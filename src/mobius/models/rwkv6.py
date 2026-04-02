# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RWKV-6 (Eagle/Finch) causal language model.

RWKV-6 is the "Finch" architecture — a linear-complexity RNN that replaces
transformer self-attention with a data-dependent WKV-6 recurrence.  Compared
to the earlier RWKV-4, it introduces:

  * **Data-dependent time mixing**: per-token mixing coefficients are computed
    via low-rank projections (time_maa_w1/w2) instead of fixed scalars.
  * **Data-dependent time decay**: per-token decay weights derived from
    tanh(x @ time_decay_w1) @ time_decay_w2 instead of a fixed vector.
  * **Matrix WKV state**: the recurrent state per layer is now a
    (num_heads, head_size, head_size) *matrix* per sample, rather than the
    scalar vectors of RWKV-4.
  * **Gate + GroupNorm** on the attention output.

Architecture per block:
    pre_ln (layer 0 only) → ln1 → time_mixing (WKV-6) → residual
                          → ln2 → channel_mixing (FFN) → residual

State per layer (single-token decode):
    shift_attn: (batch, hidden_size)                        — previous hidden for time mixing
    wkv_state:  (batch, num_heads, head_size, head_size)    — WKV-6 matrix state (float32)
    shift_ffn:  (batch, hidden_size)                        — previous hidden for channel mixing

HuggingFace reference: ``Rwkv6ForCausalLM`` (model_type="rwkv6", trust_remote_code=True).
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Rwkv6Config
from mobius.components._common import Embedding, LayerNorm, Linear

# ---------------------------------------------------------------------------
# WKV-6 time-mixing
# ---------------------------------------------------------------------------

# 5 learnable mixing-delta components (w, k, v, r, g) from a shared low-rank path.
_MIX_SPLITS = 5


class _Rwkv6TimeMixing(nn.Module):
    """RWKV-6 time-mixing (WKV-6 attention) for a single decoder layer.

    Implements the data-dependent WKV-6 recurrence for single-token decode.
    All WKV arithmetic is performed in float32.

    Parameters (match HuggingFace ``Rwkv6SelfAttention``):
        time_maa_x:       (1, 1, hidden_size)         — base x-mix coefficient
        time_maa_w/k/v/r/g: (1, 1, hidden_size)       — base per-component mix
        time_maa_w1:      (hidden_size, mix_extra*5)   — low-rank input → mix
        time_maa_w2:      (5, mix_extra, hidden_size)  — low-rank mix → delta
        time_decay:       (1, 1, attn_size)            — base log-log decay
        time_decay_w1:    (hidden_size, decay_extra)   — low-rank input → decay
        time_decay_w2:    (decay_extra, attn_size)     — low-rank decay output
        time_faaaa:       (num_heads, head_size)        — current-token bonus (u)
        key/value/receptance/gate/output: linear projections
        ln_x_weight/bias: (attn_size,)                 — GroupNorm params
    """

    def __init__(self, config: Rwkv6Config) -> None:
        super().__init__()
        H = config.hidden_size
        A = config.attention_hidden_size
        D_mix = config.time_mix_extra_dim
        D_decay = config.time_decay_extra_dim
        num_heads = A // config.head_size
        eps_gn = (config.layer_norm_epsilon) * (config.head_size_divisor**2)

        self._num_heads = num_heads
        self._head_size = config.head_size
        self._attn_size = A
        self._mix_dim = D_mix
        self._gn_eps = eps_gn

        # Base mixing coefficients — broadcast over (B, 1, H)
        self.time_maa_x = nn.Parameter([1, 1, H])
        self.time_maa_w = nn.Parameter([1, 1, H])
        self.time_maa_k = nn.Parameter([1, 1, H])
        self.time_maa_v = nn.Parameter([1, 1, H])
        self.time_maa_r = nn.Parameter([1, 1, H])
        self.time_maa_g = nn.Parameter([1, 1, H])

        # Low-rank data-dependent mixing: (H) → (5, H) per token
        self.time_maa_w1 = nn.Parameter([H, D_mix * _MIX_SPLITS])
        self.time_maa_w2 = nn.Parameter([_MIX_SPLITS, D_mix, H])

        # Per-channel base decay (log-log scale) + low-rank data-dependent part
        self.time_decay = nn.Parameter([1, 1, A])
        self.time_decay_w1 = nn.Parameter([H, D_decay])
        self.time_decay_w2 = nn.Parameter([D_decay, A])

        # Current-token bonus u = time_faaaa: (num_heads, head_size)
        self.time_faaaa = nn.Parameter([num_heads, config.head_size])

        # Projections (no bias — matches HF)
        self.receptance = Linear(H, A, bias=False)
        self.key = Linear(H, A, bias=False)
        self.value = Linear(H, A, bias=False)
        self.gate = Linear(H, A, bias=False)
        self.output = Linear(A, H, bias=False)

        # GroupNorm weight/bias (replaces nn.GroupNorm; stored as bare params
        # to avoid sub-module naming that would conflict with HF rename path)
        self.ln_x_weight = nn.Parameter([A])
        self.ln_x_bias = nn.Parameter([A])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        shift_state: ir.Value,
        wkv_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """WKV-6 time-mixing for a single token.

        Args:
            hidden:      (batch, 1, hidden_size) — current token hidden states.
            shift_state: (batch, hidden_size)    — previous hidden (time shift).
            wkv_state:   (batch, num_heads, head_size, head_size) float32.

        Returns:
            output:        (batch, 1, hidden_size) — time-mixed output.
            new_shift:     (batch, hidden_size)    — updated shift state.
            new_wkv_state: (batch, num_heads, head_size, head_size) float32.
        """
        Nh = self._num_heads
        S = self._head_size
        D_mix = self._mix_dim
        eps = self._gn_eps

        # ------------------------------------------------------------------
        # 1. Compute time-shift difference
        # ------------------------------------------------------------------
        shifted = op.Unsqueeze(shift_state, [1])  # (B, 1, H)
        xx = op.Sub(shifted, hidden)  # (B, 1, H) — delta from previous token

        # ------------------------------------------------------------------
        # 2. Data-dependent mixing deltas via low-rank projection
        # ------------------------------------------------------------------
        # Mix base: apply time_maa_x to blend hidden with shifted
        x_mix = op.Add(hidden, op.Mul(xx, self.time_maa_x))  # (B, 1, H)

        # Project → (B, 1, 5*D_mix) → tanh
        tanh_out = op.Tanh(op.MatMul(x_mix, self.time_maa_w1))  # (B, 1, 5*D)

        # Reshape to (B, 5, D_mix) then Transpose to (5, B, D_mix)
        batch = op.Shape(hidden, start=0, end=1)  # scalar
        tanh_bsd = op.Reshape(
            op.Squeeze(tanh_out, [1]),
            op.Concat(batch, op.Constant(value_ints=[_MIX_SPLITS, D_mix]), axis=0),
        )  # (B, 5, D_mix)
        tanh_5bd = op.Transpose(tanh_bsd, perm=[1, 0, 2])  # (5, B, D_mix)

        # Batch matmul: (5, B, D_mix) @ (5, D_mix, H) → (5, B, H)
        mixes = op.MatMul(tanh_5bd, self.time_maa_w2)  # (5, B, H)

        # Split into 5 per-component deltas, each (1, B, H)
        mw, mk, mv, mr, mg = op.Split(
            mixes, axis=0, num_outputs=_MIX_SPLITS, _outputs=_MIX_SPLITS
        )

        def _apply_mix(base_param, delta_1bh):
            # base_param: (1, 1, H); delta_1bh: (1, B, H) → (B, 1, H)
            delta = op.Transpose(delta_1bh, perm=[1, 0, 2])  # (B, 1, H)
            return op.Add(hidden, op.Mul(xx, op.Add(base_param, delta)))  # (B, 1, H)

        w_in = _apply_mix(self.time_maa_w, mw)   # (B, 1, H) — for decay
        k_in = _apply_mix(self.time_maa_k, mk)   # (B, 1, H)
        v_in = _apply_mix(self.time_maa_v, mv)   # (B, 1, H)
        r_in = _apply_mix(self.time_maa_r, mr)   # (B, 1, H)
        g_in = _apply_mix(self.time_maa_g, mg)   # (B, 1, H)

        # ------------------------------------------------------------------
        # 3. Linear projections + gate (SiLU = x * sigmoid(x))
        # ------------------------------------------------------------------
        r = self.receptance(op, r_in)                               # (B, 1, A)
        k = self.key(op, k_in)                                      # (B, 1, A)
        v = self.value(op, v_in)                                    # (B, 1, A)
        g_gate = self.gate(op, g_in)                                # (B, 1, A)
        g = op.Mul(g_gate, op.Sigmoid(g_gate))                      # (B, 1, A) — SiLU

        # ------------------------------------------------------------------
        # 4. Data-dependent decay
        # ------------------------------------------------------------------
        # w = time_decay + tanh(w_in @ time_decay_w1) @ time_decay_w2
        w_delta = op.MatMul(
            op.Tanh(op.MatMul(w_in, self.time_decay_w1)),  # (B, 1, D_decay)
            self.time_decay_w2,                             # (D_decay, A)
        )  # (B, 1, A)
        w = op.Add(self.time_decay, w_delta)  # (B, 1, A) — log-log decay values

        # ------------------------------------------------------------------
        # 5. WKV-6 single-token recurrence (float32)
        # ------------------------------------------------------------------
        # Squeeze seq dim since T=1
        r_sq = op.Squeeze(r, [1])  # (B, A)
        k_sq = op.Squeeze(k, [1])  # (B, A)
        v_sq = op.Squeeze(v, [1])  # (B, A)
        w_sq = op.Squeeze(w, [1])  # (B, A) — log-log decay

        # Cast to float32 for numerically stable WKV
        r_f32 = op.Cast(r_sq, to=ir.DataType.FLOAT)  # (B, A)
        k_f32 = op.Cast(k_sq, to=ir.DataType.FLOAT)  # (B, A)
        v_f32 = op.Cast(v_sq, to=ir.DataType.FLOAT)  # (B, A)
        w_f32 = op.Cast(w_sq, to=ir.DataType.FLOAT)  # (B, A)

        # Reshape to multi-head layout: (B, Nh, S, ?)
        attn_shape = op.Constant(value_ints=[Nh, S])
        r_heads = op.Reshape(
            r_f32, op.Concat(batch, op.Constant(value_ints=[Nh, 1, S]), axis=0)
        )  # (B, Nh, 1, S) — row vector per head
        k_heads = op.Reshape(
            k_f32, op.Concat(batch, op.Constant(value_ints=[Nh, S, 1]), axis=0)
        )  # (B, Nh, S, 1) — col vector per head (for outer product)
        v_heads = op.Reshape(
            v_f32, op.Concat(batch, op.Constant(value_ints=[Nh, 1, S]), axis=0)
        )  # (B, Nh, 1, S) — row vector per head

        # Decay per step: actual_w = exp(-exp(w_log_log))
        # Reshape to (B, Nh, S, 1) for broadcasting with state (B, Nh, S, S)
        w_decay = op.Reshape(
            op.Exp(op.Neg(op.Exp(w_f32))),
            op.Concat(batch, op.Constant(value_ints=[Nh, S, 1]), axis=0),
        )  # (B, Nh, S, 1)

        # Outer product: kv = k_heads @ v_heads = (B, Nh, S, 1) @ (B, Nh, 1, S) = (B, Nh, S, S)
        kv = op.MatMul(k_heads, v_heads)  # (B, Nh, S, S)

        # Current-token bonus: time_faaaa (Nh, S) → (Nh, S, 1) for broadcasting
        u = op.Reshape(self.time_faaaa, op.Constant(value_ints=[Nh, S, 1]))  # (Nh, S, 1)

        # Apply bonus and add previous state:
        # attn_bonus = u * kv + state
        attn_bonus = op.Add(op.Mul(u, kv), wkv_state)  # (B, Nh, S, S)

        # Receptance-gated output: r @ attn_bonus = (B, Nh, 1, S) @ (B, Nh, S, S) = (B, Nh, 1, S)
        out_heads = op.MatMul(r_heads, attn_bonus)  # (B, Nh, 1, S)

        # Update state: new_state = kv + decay * state
        new_wkv_state = op.Add(kv, op.Mul(w_decay, wkv_state))  # (B, Nh, S, S)

        # ------------------------------------------------------------------
        # 6. GroupNorm (normalize over S within each head)
        # ------------------------------------------------------------------
        # Reshape (B, Nh, 1, S) → (B, Nh, S) → compute mean/var over S dim
        out_bhs = op.Reshape(
            out_heads, op.Concat(batch, op.Constant(value_ints=[Nh, S]), axis=0)
        )  # (B, Nh, S)
        mean = op.ReduceMean(out_bhs, axes=[-1], keepdims=1)  # (B, Nh, 1)
        var = op.ReduceMean(
            op.Pow(op.Sub(out_bhs, mean), op.CastLike(op.Constant(value_float=2.0), out_bhs)),
            axes=[-1],
            keepdims=1,
        )  # (B, Nh, 1)
        out_norm = op.Div(
            op.Sub(out_bhs, mean),
            op.Sqrt(op.Add(var, op.CastLike(op.Constant(value_float=eps), out_bhs))),
        )  # (B, Nh, S)

        # Flatten to (B, A) and apply weight/bias
        out_flat = op.Reshape(
            out_norm, op.Concat(batch, op.Constant(value_ints=[-1]), axis=0)
        )  # (B, A)
        ln_w = op.CastLike(self.ln_x_weight, out_flat)
        ln_b = op.CastLike(self.ln_x_bias, out_flat)
        out_scaled = op.Add(op.Mul(out_flat, ln_w), ln_b)  # (B, A)

        # ------------------------------------------------------------------
        # 7. Gate and output projection
        # ------------------------------------------------------------------
        g_sq = op.Squeeze(g, [1])  # (B, A)
        # Cast back to model dtype before gating
        out_gated = op.CastLike(
            op.Mul(out_scaled, op.Cast(g_sq, to=ir.DataType.FLOAT)),
            hidden,
        )  # (B, A)

        out_unsqueeze = op.Unsqueeze(out_gated, [1])  # (B, 1, A)
        output = self.output(op, out_unsqueeze)  # (B, 1, H)

        # New shift state = current hidden (squeezed)
        new_shift = op.Squeeze(hidden, [1])  # (B, H)

        return output, new_shift, new_wkv_state


# ---------------------------------------------------------------------------
# Channel-mixing (FFN)
# ---------------------------------------------------------------------------


class _Rwkv6ChannelMixing(nn.Module):
    """RWKV-6 channel-mixing (feed-forward).

    Identical structure to RWKV-4 channel mixing with updated parameter names:
        output = sigmoid(receptance(mixed_r)) * value(relu(key(mixed_k))^2)

    Parameters (match HuggingFace ``Rwkv6FeedForward``):
        time_maa_k:   (1, 1, hidden_size)
        time_maa_r:   (1, 1, hidden_size)
        key.weight:   (intermediate_size, hidden_size)
        receptance.weight: (hidden_size, hidden_size)
        value.weight: (hidden_size, intermediate_size)
    """

    def __init__(self, config: Rwkv6Config) -> None:
        super().__init__()
        H = config.hidden_size
        I = config.intermediate_size

        self.time_maa_k = nn.Parameter([1, 1, H])
        self.time_maa_r = nn.Parameter([1, 1, H])

        self.key = Linear(H, I, bias=False)
        self.receptance = Linear(H, H, bias=False)
        self.value = Linear(I, H, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        shift_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Channel mixing for a single token.

        Args:
            hidden:      (batch, 1, hidden_size) — input.
            shift_state: (batch, hidden_size)    — previous hidden (time shift).

        Returns:
            output:    (batch, 1, hidden_size) — channel-mixed output.
            new_shift: (batch, hidden_size)    — updated shift state.
        """
        shifted = op.Unsqueeze(shift_state, [1])  # (B, 1, H)
        xx = op.Sub(shifted, hidden)  # (B, 1, H)

        k_in = op.Add(hidden, op.Mul(xx, self.time_maa_k))  # (B, 1, H)
        r_in = op.Add(hidden, op.Mul(xx, self.time_maa_r))  # (B, 1, H)

        # Squared ReLU key transform
        k = self.key(op, k_in)  # (B, 1, I)
        k_relu = op.Relu(k)
        k = op.Mul(k_relu, k_relu)  # (B, 1, I) — squared relu

        # Sigmoid gating
        r = op.Sigmoid(self.receptance(op, r_in))  # (B, 1, H)

        output = op.Mul(r, self.value(op, k))  # (B, 1, H)

        # New shift = current hidden
        new_shift = op.Squeeze(hidden, [1])  # (B, H)

        return output, new_shift


# ---------------------------------------------------------------------------
# RWKV-6 block
# ---------------------------------------------------------------------------


class _Rwkv6Block(nn.Module):
    """Single RWKV-6 decoder block.

    Structure:
        pre_ln (layer 0 only) → ln1 → time_mixing → residual
                              → ln2 → channel_mixing → residual
    """

    def __init__(self, config: Rwkv6Config, is_first_layer: bool = False) -> None:
        super().__init__()
        self._is_first = is_first_layer

        if is_first_layer:
            self.pre_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attention = _Rwkv6TimeMixing(config)
        self.feed_forward = _Rwkv6ChannelMixing(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        layer_state: tuple[ir.Value, ir.Value, ir.Value],
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value, ir.Value]]:
        """Forward pass for one RWKV-6 block.

        Args:
            hidden:      (batch, 1, hidden_size) — input hidden states.
            layer_state: (shift_attn, wkv_state, shift_ffn)

        Returns:
            output:    (batch, 1, hidden_size)
            new_state: updated (shift_attn, wkv_state, shift_ffn)
        """
        shift_attn, wkv_state, shift_ffn = layer_state

        if self._is_first:
            hidden = self.pre_ln(op, hidden)

        # Time mixing with residual
        attn_out, new_shift_attn, new_wkv = self.attention(
            op, self.ln1(op, hidden), shift_attn, wkv_state
        )
        hidden = op.Add(hidden, attn_out)

        # Channel mixing with residual
        ffn_out, new_shift_ffn = self.feed_forward(op, self.ln2(op, hidden), shift_ffn)
        hidden = op.Add(hidden, ffn_out)

        return hidden, (new_shift_attn, new_wkv, new_shift_ffn)


# ---------------------------------------------------------------------------
# Full RWKV-6 causal LM model
# ---------------------------------------------------------------------------

_LayerState6 = tuple[ir.Value, ir.Value, ir.Value]


class Rwkv6CausalLMModel(nn.Module):
    """RWKV-6 causal language model (single-token decode mode).

    Processes one token per forward call with recurrent per-layer state.
    Compatible with HuggingFace ``Rwkv6ForCausalLM`` (trust_remote_code=True).

    Inputs:
        input_ids:   (batch, 1) INT64
        past_states: per-layer (shift_attn, wkv_state, shift_ffn)

    Outputs:
        logits:         (batch, 1, vocab_size)
        present_states: updated per-layer state tuples

    Weight names match HuggingFace after stripping ``rwkv6.`` prefix and
    renaming ``attention.ln_x.weight/bias`` → ``attention.ln_x_weight/bias``.
    """

    def __init__(self, config: Rwkv6Config) -> None:
        super().__init__()
        self._config = config

        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                _Rwkv6Block(config, is_first_layer=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_out = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[_LayerState6] | None = None,
    ) -> tuple[ir.Value, list[_LayerState6]]:
        """Single-token forward pass.

        Args:
            input_ids:   (batch, 1) INT64.
            past_states: Per-layer (shift_attn, wkv_state, shift_ffn); None on first call.

        Returns:
            logits:         (batch, 1, vocab_size)
            present_states: updated per-layer state tuples
        """
        hidden = self.embeddings(op, input_ids)  # (B, 1, H)

        present_states: list[_LayerState6] = []
        past = past_states or [None] * len(self.blocks)  # type: ignore[list-item]

        for block, state in zip(self.blocks, past):
            hidden, new_state = block(op, hidden, state)
            present_states.append(new_state)

        hidden = self.ln_out(op, hidden)  # (B, 1, H)
        logits = self.head(op, hidden)    # (B, 1, V)

        return logits, present_states

    def preprocess_weights(self, state_dict: dict[str, object]) -> dict[str, object]:
        """Map HuggingFace weight names to ONNX module names.

        Steps:
        1. Strip ``rwkv6.`` outer prefix.
        2. Rename ``attention.ln_x.weight/bias`` → ``attention.ln_x_weight/bias``
           (GroupNorm stored as bare params in our module).
        3. Apply ``rescale_every`` weight scaling so inference doesn't divide
           hidden states (same convention as RWKV-4).

        Representative name mappings:
            rwkv6.embeddings.weight              → embeddings.weight
            rwkv6.blocks.N.attention.time_maa_x  → blocks.N.attention.time_maa_x
            rwkv6.blocks.N.attention.ln_x.weight → blocks.N.attention.ln_x_weight
            rwkv6.blocks.N.attention.ln_x.bias   → blocks.N.attention.ln_x_bias
            rwkv6.ln_out.weight                  → ln_out.weight
            head.weight                          → head.weight
        """
        import re

        config = self._config
        rescale_every = config.rescale_every

        result: dict[str, object] = {}
        for key, value in state_dict.items():
            k = key
            # Strip outer model prefix
            if k.startswith("rwkv6."):
                k = k[len("rwkv6."):]
            # GroupNorm sub-module → bare param rename
            k = re.sub(r"\.ln_x\.(weight|bias)$", r".ln_x_\1", k)
            result[k] = value

        # Bake rescale_every scaling into weights
        if rescale_every > 0:
            for block_id in range(config.num_hidden_layers):
                if (block_id + 1) % rescale_every == 0:
                    scale = float(2 ** ((block_id + 1) // rescale_every))
                    for wname in (
                        f"blocks.{block_id}.attention.output.weight",
                        f"blocks.{block_id}.feed_forward.value.weight",
                    ):
                        if wname in result:
                            import numpy as np  # noqa: PLC0415

                            w = result[wname]
                            try:
                                import torch  # noqa: PLC0415

                                if isinstance(w, torch.Tensor):
                                    result[wname] = w * scale
                                    continue
                            except ImportError:
                                pass
                            result[wname] = np.array(w, dtype=np.float32) * scale

        return result
