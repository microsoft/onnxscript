# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RWKV (Receptance Weighted Key Value) causal language model.

RWKV is a linear-complexity RNN that replaces transformer attention with a
time-mixing (WKV) recurrence and a channel-mixing feed-forward.  Unlike
transformers, it uses **no** attention mask and carries a fixed-size state
per layer instead of a growing KV cache.

Architecture per block:
    pre_ln (layer 0 only) → time_mixing → residual
    ln2 → channel_mixing → residual

Time-mixing (WKV attention):
    key, value, receptance are linear projections of a time-shifted input.
    The WKV recurrence accumulates a weighted sum of past values:
        output_t = receptance * (num / den)
    where num/den are updated with an exponential decay per step.

Channel-mixing (Feed-forward):
    key, receptance are linear projections of a time-shifted input.
    output = receptance * value(relu(key)^2)

State per layer (single-token decode):
    shift_attn: (batch, hidden_size)          — previous hidden for time mixing
    wkv_num:    (batch, attention_hidden_size) — WKV numerator accumulator
    wkv_den:    (batch, attention_hidden_size) — WKV denominator accumulator
    wkv_max:    (batch, attention_hidden_size) — WKV max (numerical stability)
    shift_ffn:  (batch, hidden_size)          — previous hidden for channel mixing

HuggingFace reference: ``RwkvForCausalLM`` (model_type="rwkv").
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import RwkvConfig
from mobius.components._common import Embedding, LayerNorm, Linear

# ---------------------------------------------------------------------------
# Time-mixing (WKV attention)
# ---------------------------------------------------------------------------


class _RwkvTimeMixing(nn.Module):
    """RWKV time-mixing (WKV attention) for a single decoder layer.

    Implements the WKV recurrent attention using the numerically stable
    log-sum-exp formulation.  All WKV arithmetic is performed in float32
    and cast back to the model dtype on output.

    Parameters (match HuggingFace ``RwkvSelfAttention``):
        time_decay:          (attention_hidden_size,) — log of per-dim decay
        time_first:          (attention_hidden_size,) — current-token bonus
        time_mix_key:        (1, 1, hidden_size)      — key interpolation coefficient
        time_mix_value:      (1, 1, hidden_size)      — value interpolation coefficient
        time_mix_receptance: (1, 1, hidden_size)      — receptance interpolation coefficient
        key.weight:          (attention_hidden_size, hidden_size)
        value.weight:        (attention_hidden_size, hidden_size)
        receptance.weight:   (attention_hidden_size, hidden_size)
        output.weight:       (hidden_size, attention_hidden_size)
    """

    def __init__(self, hidden_size: int, attention_hidden_size: int):
        super().__init__()
        self._attn_size = attention_hidden_size

        # Learnable decay: time_decay is log-scale; we negate+exp at forward time.
        self.time_decay = nn.Parameter([attention_hidden_size])
        # Current-token bonus added to key before WKV (u in the RWKV paper).
        self.time_first = nn.Parameter([attention_hidden_size])

        # Mixing coefficients for key, value, receptance (1,1,H for broadcast).
        self.time_mix_key = nn.Parameter([1, 1, hidden_size])
        self.time_mix_value = nn.Parameter([1, 1, hidden_size])
        self.time_mix_receptance = nn.Parameter([1, 1, hidden_size])

        # Linear projections (no bias, matching HF).
        self.key = Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = Linear(attention_hidden_size, hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        shift_state: ir.Value,
        wkv_num: ir.Value,
        wkv_den: ir.Value,
        wkv_max: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
        """Apply WKV time-mixing for a single token.

        Args:
            hidden:      (batch, 1, hidden_size) — current token representation.
            shift_state: (batch, hidden_size)    — previous hidden (time shift).
            wkv_num:     (batch, attention_hidden_size) — WKV numerator state.
            wkv_den:     (batch, attention_hidden_size) — WKV denominator state.
            wkv_max:     (batch, attention_hidden_size) — WKV max state.

        Returns:
            output:       (batch, 1, hidden_size) — time-mixed output.
            new_shift:    (batch, hidden_size)    — updated shift state (= current hidden).
            new_wkv_num:  (batch, attention_hidden_size) — updated numerator.
            new_wkv_den:  (batch, attention_hidden_size) — updated denominator.
            new_wkv_max:  (batch, attention_hidden_size) — updated max.
        """
        # Expand shift state to (batch, 1, hidden_size) for mixing.
        shifted = op.Unsqueeze(shift_state, [1])  # (B, 1, H)

        # Time mixing: x = hidden * mix + prev * (1 - mix)
        one = op.CastLike(op.Constant(value_float=1.0), hidden)
        key_in = op.Add(
            op.Mul(hidden, self.time_mix_key),
            op.Mul(shifted, op.Sub(one, self.time_mix_key)),
        )  # (B, 1, H)
        val_in = op.Add(
            op.Mul(hidden, self.time_mix_value),
            op.Mul(shifted, op.Sub(one, self.time_mix_value)),
        )  # (B, 1, H)
        rec_in = op.Add(
            op.Mul(hidden, self.time_mix_receptance),
            op.Mul(shifted, op.Sub(one, self.time_mix_receptance)),
        )  # (B, 1, H)

        # Project to attention hidden size and squeeze seq dim.
        k = op.Squeeze(self.key(op, key_in), [1])  # (B, attn_h)
        v = op.Squeeze(self.value(op, val_in), [1])  # (B, attn_h)
        r = op.Sigmoid(op.Squeeze(self.receptance(op, rec_in), [1]))  # (B, attn_h)

        # ── WKV computation in float32 ──────────────────────────────────────
        # time_decay_neg = -exp(time_decay): negative per-dim decay.
        time_decay_neg = op.Neg(
            op.Exp(op.Cast(self.time_decay, to=ir.DataType.FLOAT))
        )  # (attn_h,)

        k_f32 = op.Cast(k, to=ir.DataType.FLOAT)  # (B, attn_h)
        v_f32 = op.Cast(v, to=ir.DataType.FLOAT)  # (B, attn_h)
        num_f32 = op.Cast(wkv_num, to=ir.DataType.FLOAT)
        den_f32 = op.Cast(wkv_den, to=ir.DataType.FLOAT)
        max_f32 = op.Cast(wkv_max, to=ir.DataType.FLOAT)
        time_first_f32 = op.Cast(self.time_first, to=ir.DataType.FLOAT)  # (attn_h,)

        # Current token output: use time_first as bonus key weight.
        # max_o = max(max_state, k + time_first)
        k_plus_u = op.Add(k_f32, time_first_f32)  # (B, attn_h)
        max_o = op.Max(max_f32, k_plus_u)
        e1 = op.Exp(op.Sub(max_f32, max_o))  # (B, attn_h)
        e2 = op.Exp(op.Sub(k_plus_u, max_o))  # (B, attn_h)
        wkv = op.Div(
            op.Add(op.Mul(e1, num_f32), op.Mul(e2, v_f32)),
            op.Add(op.Mul(e1, den_f32), e2),
        )  # (B, attn_h)

        # Apply receptance gate and output projection.
        attn_out = op.CastLike(op.Mul(op.Cast(r, to=ir.DataType.FLOAT), wkv), k)
        attn_out = op.Unsqueeze(attn_out, [1])  # (B, 1, attn_h)
        output = self.output(op, attn_out)  # (B, 1, hidden_size)

        # ── State update ────────────────────────────────────────────────────
        # Decay previous state and add contribution of current key.
        # max_s = max(max_state + decay, k)
        max_s = op.Max(op.Add(max_f32, time_decay_neg), k_f32)  # (B, attn_h)
        e1_s = op.Exp(op.Sub(op.Add(max_f32, time_decay_neg), max_s))
        e2_s = op.Exp(op.Sub(k_f32, max_s))
        new_num = op.Add(op.Mul(e1_s, num_f32), op.Mul(e2_s, v_f32))
        new_den = op.Add(op.Mul(e1_s, den_f32), e2_s)

        # New shift state = current hidden (squeeze seq dim).
        new_shift = op.Squeeze(hidden, [1])  # (B, H)

        return (
            output,
            new_shift,
            op.CastLike(new_num, wkv_num),
            op.CastLike(new_den, wkv_den),
            op.CastLike(max_s, wkv_max),
        )


# ---------------------------------------------------------------------------
# Channel-mixing (Feed-forward)
# ---------------------------------------------------------------------------


class _RwkvChannelMixing(nn.Module):
    """RWKV channel-mixing (feed-forward) for a single decoder layer.

    Implements the channel mixing operation:
        output = receptance * value(relu(key(mixed_x))^2)

    Parameters (match HuggingFace ``RwkvFeedForward``):
        time_mix_key:        (1, 1, hidden_size)
        time_mix_receptance: (1, 1, hidden_size)
        key.weight:          (intermediate_size, hidden_size)
        receptance.weight:   (hidden_size, hidden_size)
        value.weight:        (hidden_size, intermediate_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.time_mix_key = nn.Parameter([1, 1, hidden_size])
        self.time_mix_receptance = nn.Parameter([1, 1, hidden_size])

        self.key = Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = Linear(hidden_size, hidden_size, bias=False)
        self.value = Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        shift_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Apply channel mixing for a single token.

        Args:
            hidden:      (batch, 1, hidden_size) — input.
            shift_state: (batch, hidden_size)    — previous hidden (time shift).

        Returns:
            output:    (batch, 1, hidden_size) — channel-mixed output.
            new_shift: (batch, hidden_size)    — updated shift state.
        """
        shifted = op.Unsqueeze(shift_state, [1])  # (B, 1, H)

        one = op.CastLike(op.Constant(value_float=1.0), hidden)
        key_in = op.Add(
            op.Mul(hidden, self.time_mix_key),
            op.Mul(shifted, op.Sub(one, self.time_mix_key)),
        )
        rec_in = op.Add(
            op.Mul(hidden, self.time_mix_receptance),
            op.Mul(shifted, op.Sub(one, self.time_mix_receptance)),
        )

        # key: squared relu (RWKV's nonlinearity)
        k = self.key(op, key_in)  # (B, 1, intermediate_size)
        k = op.Mul(op.Relu(k), op.Relu(k))

        # Receptance gate (sigmoid)
        r = op.Sigmoid(self.receptance(op, rec_in))  # (B, 1, H)

        # Gated output
        output = op.Mul(r, self.value(op, k))  # (B, 1, H)

        # New shift state = current hidden
        new_shift = op.Squeeze(hidden, [1])  # (B, H)

        return output, new_shift


# ---------------------------------------------------------------------------
# RWKV block (single layer)
# ---------------------------------------------------------------------------


class _RwkvBlock(nn.Module):
    """Single RWKV decoder block.

    Structure:
        pre_ln (layer 0 only) → ln1 → time_mixing → residual
                               → ln2 → channel_mixing → residual

    The ``pre_ln`` is only present in the first layer (layer_id=0) and is
    applied to the token embeddings before the first block.

    HuggingFace reference: ``RwkvBlock``.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_hidden_size: int,
        intermediate_size: int,
        layer_norm_epsilon: float,
        is_first_layer: bool = False,
    ):
        super().__init__()
        self._is_first = is_first_layer

        if is_first_layer:
            # Pre-norm applied to raw embeddings in the first layer only.
            self.pre_ln = LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.ln1 = LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.ln2 = LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.attention = _RwkvTimeMixing(hidden_size, attention_hidden_size)
        self.feed_forward = _RwkvChannelMixing(hidden_size, intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        layer_state: tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value],
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]]:
        """Forward pass for one RWKV block.

        Args:
            hidden:       (batch, 1, hidden_size) — input hidden states.
            layer_state:  (shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn)

        Returns:
            output:       (batch, 1, hidden_size)
            new_state:    updated (shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn)
        """
        shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn = layer_state

        # Pre-LN only in first layer
        if self._is_first:
            hidden = self.pre_ln(op, hidden)

        # Time mixing with residual
        attn_out, new_shift_attn, new_num, new_den, new_max = self.attention(
            op, self.ln1(op, hidden), shift_attn, wkv_num, wkv_den, wkv_max
        )
        hidden = op.Add(hidden, attn_out)

        # Channel mixing with residual
        ffn_out, new_shift_ffn = self.feed_forward(op, self.ln2(op, hidden), shift_ffn)
        hidden = op.Add(hidden, ffn_out)

        return hidden, (new_shift_attn, new_num, new_den, new_max, new_shift_ffn)


# ---------------------------------------------------------------------------
# Full RWKV causal LM model
# ---------------------------------------------------------------------------

# Type alias for per-layer state.
_LayerState = tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]


class RwkvCausalLMModel(nn.Module):
    """RWKV causal language model (single-token decode mode).

    Processes one token per forward call, carrying recurrent state across
    calls.  The state layout matches HuggingFace ``RwkvForCausalLM``.

    Inputs:
        input_ids:   (batch, 1)          — current token ID.
        past_states: list of per-layer state tuples
            (shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn)

    Outputs:
        logits:      (batch, 1, vocab_size)
        present_states: updated per-layer state tuples

    Weight names map directly to HuggingFace after stripping ``rwkv.``
    prefix and applying rescale_every scaling.
    """

    def __init__(self, config: RwkvConfig):
        super().__init__()
        self._config = config
        hidden_size = config.hidden_size
        attn_size = config.attention_hidden_size
        intermediate_size = config.intermediate_size
        eps = config.layer_norm_epsilon

        self.embeddings = Embedding(config.vocab_size, hidden_size)

        self.blocks = nn.ModuleList(
            [
                _RwkvBlock(
                    hidden_size=hidden_size,
                    attention_hidden_size=attn_size,
                    intermediate_size=intermediate_size,
                    layer_norm_epsilon=eps,
                    is_first_layer=(i == 0),
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.ln_out = LayerNorm(hidden_size, eps=eps)
        # LM head (not tied by default in RWKV)
        self.head = Linear(hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[_LayerState] | None = None,
    ) -> tuple[ir.Value, list[_LayerState]]:
        """Single-token forward pass.

        Args:
            input_ids:   (batch, 1) INT64.
            past_states: Per-layer state tuples; None on the first call.

        Returns:
            logits:         (batch, 1, vocab_size)
            present_states: Updated per-layer state tuples.
        """
        # Embed: (B, 1) → (B, 1, H)
        hidden = self.embeddings(op, input_ids)

        present_states: list[_LayerState] = []
        past = past_states or [None] * len(self.blocks)  # type: ignore[list-item]

        for block, state in zip(self.blocks, past):
            hidden, new_state = block(op, hidden, state)
            present_states.append(new_state)

        hidden = self.ln_out(op, hidden)  # (B, 1, H)
        logits = self.head(op, hidden)  # (B, 1, V)

        return logits, present_states

    def preprocess_weights(self, state_dict: dict[str, object]) -> dict[str, object]:
        """Map HuggingFace weight names to ONNX module names.

        Steps:
        1. Strip ``rwkv.`` outer prefix.
        2. Apply rescale_every weight scaling so the ONNX model can run
           inference without dividing hidden states at each layer.

        rescale_every bakes a 2^(block_id // rescale_every) scale into
        ``blocks.N.attention.output.weight`` and
        ``blocks.N.feed_forward.value.weight`` for inference.
        Only applied if the weights have NOT already been rescaled
        (HF checkpoints are typically saved in training-mode).

        HF name → ONNX name (representative examples):
            rwkv.embeddings.weight          → embeddings.weight
            rwkv.blocks.N.attention.*       → blocks.N.attention.*
            rwkv.blocks.N.feed_forward.*    → blocks.N.feed_forward.*
            rwkv.ln_out.*                   → ln_out.*
            head.weight                     → head.weight
        """
        import torch

        config = self._config
        rescale_every = config.rescale_every

        result: dict[str, object] = {}
        for key, value in state_dict.items():
            k = key
            # Strip outer model prefix
            if k.startswith("rwkv."):
                k = k[len("rwkv.") :]
            result[k] = value

        # Apply rescale_every: scale attention output and FFN value weights
        # so the ONNX forward does NOT need to divide hidden states.
        if rescale_every > 0:
            for block_id in range(config.num_hidden_layers):
                if (block_id + 1) % rescale_every == 0:
                    scale = float(2 ** ((block_id + 1) // rescale_every))
                    for wname in (
                        f"blocks.{block_id}.attention.output.weight",
                        f"blocks.{block_id}.feed_forward.value.weight",
                    ):
                        if wname in result:
                            w = result[wname]
                            if hasattr(w, "mul_") and isinstance(w, torch.Tensor):
                                result[wname] = w * scale
                            else:
                                import numpy as np

                                result[wname] = np.array(w, dtype=np.float32) * scale

        return result
