# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Falcon and Bloom CausalLM models.

Falcon supports two positional encoding schemes:
- ALiBi: Linear distance-based attention bias (Falcon-40B, 180B, Bloom)
- RoPE: Rotary position embeddings (Falcon-7B, Falcon-2)

Falcon also supports parallel attention+MLP (default) or sequential:
    attention_score[i,j] += -slope * |i - j|  (ALiBi only)

HF weight naming:
- Falcon: transformer.h.N.self_attention.{query_key_value,dense}.weight
- Falcon: transformer.h.N.mlp.{dense_h_to_4h,dense_4h_to_h}.weight
- Bloom: transformer.h.N.self_attention.{query_key_value,dense}.weight
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv
from mobius.components import (
    Attention,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._common import Embedding, LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir
    from onnxscript._internal import builder


class _ALiBiAttention(nn.Module):
    """Self-attention with ALiBi positional bias."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attn_o_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple | None = None,
    ):
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        attn_out, present_key, present_value = op.Attention(
            q,
            k,
            v,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_kv_heads,
            scale=float(self.head_dim**-0.5),
            is_causal=0,  # ALiBi bias already includes causal mask
            _outputs=3,
        )

        attn_out = self.o_proj(op, attn_out)
        return attn_out, (present_key, present_value)


class _FalconMLP(nn.Module):
    """Simple 2-layer MLP with GELU activation (no gating).

    Matches HF FalconMLP: dense_h_to_4h → GELU → dense_4h_to_h.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size, bias=bias)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.dense_h_to_4h(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        return self.dense_4h_to_h(op, hidden_states)


class _ALiBiDecoderLayer(nn.Module):
    """Decoder layer with ALiBi attention and parallel/sequential support.

    When parallel_attn=True (Falcon-40B/180B), attention and MLP are computed
    in parallel and their outputs are summed before the residual addition.

    Attribute names match HF Falcon naming (ln_attn, ln_mlp, self_attention).
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        parallel_attn: bool,
        dual_ln: bool,
    ):
        super().__init__()
        self._parallel_attn = parallel_attn
        self._dual_ln = dual_ln
        self.ln_attn = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not parallel_attn or dual_ln:
            self.ln_mlp = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = _ALiBiAttention(config)
        self.mlp = _FalconMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        attn_ln_out = self.ln_attn(op, hidden_states)

        attn_out, present_kv = self.self_attention(
            op,
            attn_ln_out,
            attention_bias,
            past_key_value,
        )

        if self._parallel_attn:
            # Parallel: MLP input from separate LN or shared LN output
            if self._dual_ln:
                mlp_ln_out = self.ln_mlp(op, hidden_states)
            else:
                mlp_ln_out = attn_ln_out
            mlp_out = self.mlp(op, mlp_ln_out)
            hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        else:
            # Sequential: standard pre-norm pattern
            hidden_states = op.Add(residual, attn_out)
            residual = hidden_states
            hidden_states = self.ln_mlp(op, hidden_states)
            hidden_states = self.mlp(op, hidden_states)
            hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _FalconDecoderLayer(nn.Module):
    """Decoder layer with RoPE attention and parallel/sequential support.

    Used by Falcon-7B and Falcon-2 which use RoPE instead of ALiBi.
    In parallel mode (default), attention and MLP are computed in parallel.

    Attribute names match HF Falcon naming (ln_attn, ln_mlp, self_attention).
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        parallel_attn: bool,
        dual_ln: bool,
    ):
        super().__init__()
        self._parallel_attn = parallel_attn
        self._dual_ln = dual_ln
        self.ln_attn = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not parallel_attn or dual_ln:
            self.ln_mlp = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = Attention(config)
        self.mlp = _FalconMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        attn_ln_out = self.ln_attn(op, hidden_states)

        attn_out, present_kv = self.self_attention(
            op,
            attn_ln_out,
            attention_bias,
            position_embeddings,
            past_key_value,
        )

        if self._parallel_attn:
            if self._dual_ln:
                mlp_ln_out = self.ln_mlp(op, hidden_states)
            else:
                mlp_ln_out = attn_ln_out
            mlp_out = self.mlp(op, mlp_ln_out)
            hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        else:
            hidden_states = op.Add(residual, attn_out)
            residual = hidden_states
            hidden_states = self.ln_mlp(op, hidden_states)
            hidden_states = self.mlp(op, hidden_states)
            hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


def _create_alibi_bias(op, num_heads: int, seq_len, total_len):
    """Create ALiBi attention bias tensor.

    Returns attention bias of shape [1, num_heads, seq_len, total_len]
    where bias[0, h, i, j] = -slope_h * |position_i - position_j|.
    Slopes are geometrically spaced: 2^(-8/num_heads * (h+1)).
    """
    # Compute slopes
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = []
    for i in range(1, closest_power_of_2 + 1):
        slopes.append(base**i)
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining = min(closest_power_of_2, num_heads - closest_power_of_2)
        for i in range(1, 2 * num_remaining + 1, 2):
            slopes.append(extra_base**i)

    slopes_np = np.array(slopes[:num_heads], dtype=np.float32)
    slopes_const = op.Constant(value_floats=slopes_np.tolist())  # [num_heads]

    # Position indices: query positions [0..seq_len-1], kv positions [0..total_len-1]
    # Distance: q_pos - kv_pos (negative for past positions = causal)
    # For causal: use q_pos - kv_pos, mask future with -inf

    # Create bias as: -slope * abs(q_idx - kv_idx), then mask future
    q_indices = op.Cast(
        op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len, [0]),
            op.Constant(value_int=1),
        ),
        to=1,
    )  # [seq_len]

    kv_indices = op.Cast(
        op.Range(
            op.Constant(value_int=0),
            op.Squeeze(total_len, [0]),
            op.Constant(value_int=1),
        ),
        to=1,
    )  # [total_len]

    # Distance: [seq_len, 1] - [1, total_len] → [seq_len, total_len]
    q_expanded = op.Unsqueeze(q_indices, [1])
    kv_expanded = op.Unsqueeze(kv_indices, [0])

    # For autoregressive: offset q_indices by (total_len - seq_len)
    offset = op.Sub(op.Squeeze(total_len, [0]), op.Squeeze(seq_len, [0]))
    offset_f = op.Cast(offset, to=1)
    q_with_offset = op.Add(q_expanded, op.Unsqueeze(offset_f, [0, 1]))
    distance = op.Sub(q_with_offset, kv_expanded)  # [seq_len, total_len]

    # ALiBi: -slope * distance (positive distance = past = negative bias)
    neg_distance = op.Neg(op.Abs(distance))  # [seq_len, total_len]

    # Broadcast: slopes [num_heads] → [1, num_heads, 1, 1]
    slopes_4d = op.Unsqueeze(slopes_const, [0, 2, 3])
    bias_2d = op.Unsqueeze(neg_distance, [0, 1])  # [1, 1, seq_len, total_len]
    alibi = op.Mul(slopes_4d, bias_2d)  # [1, num_heads, seq_len, total_len]

    # Causal mask: mask future positions with large negative value
    causal_mask = op.Where(
        op.GreaterOrEqual(q_with_offset, kv_expanded),
        op.Constant(value_float=0.0),
        op.Constant(value_float=-10000.0),
    )  # [seq_len, total_len]
    causal_4d = op.Unsqueeze(causal_mask, [0, 1])  # [1, 1, seq_len, total_len]
    return op.Add(alibi, causal_4d)


class _FalconTextModel(nn.Module):
    """Falcon text model supporting both ALiBi and RoPE positional encoding.

    Attribute names match HF Falcon naming (word_embeddings, h, ln_f).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._alibi = config.alibi
        self._dtype = config.dtype
        self._num_heads = config.num_attention_heads

        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)

        parallel_attn = config.parallel_attn
        # Falcon new_decoder_architecture implies dual LN for parallel mode
        # Detect via: if num_kv_heads != num_attention_heads and alibi,
        # it's likely Falcon-40B/180B with new_decoder_architecture.
        # For test configs, default to False (single LN).
        dual_ln = False

        self.h = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            if self._alibi:
                self.h.append(_ALiBiDecoderLayer(config, parallel_attn, dual_ln))
            else:
                self.h.append(_FalconDecoderLayer(config, parallel_attn, dual_ln))
        self.ln_f = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not self._alibi:
            self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.word_embeddings(op, input_ids)
        present_key_values = []

        if self._alibi:
            # ALiBi path: compute bias once for all layers
            seq_len = op.Shape(input_ids, start=1, end=2)
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = op.Shape(past_key_values[0][0], start=2, end=3)
                total_len = op.Add(seq_len, past_len)
            else:
                total_len = seq_len

            alibi_bias = _create_alibi_bias(op, self._num_heads, seq_len, total_len)

            for i, layer in enumerate(self.h):
                past_kv = past_key_values[i] if past_key_values else None
                hidden_states, present_kv = layer(
                    op,
                    hidden_states,
                    alibi_bias,
                    past_kv,
                )
                present_key_values.append(present_kv)
        else:
            # RoPE path: compute position embeddings and attention bias
            position_embeddings = self.rotary_emb(op, position_ids)
            attention_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )

            for i, layer in enumerate(self.h):
                past_kv = past_key_values[i] if past_key_values else None
                hidden_states, present_kv = layer(
                    op,
                    hidden_states,
                    attention_bias,
                    position_embeddings,
                    past_kv,
                )
                present_key_values.append(present_kv)

        hidden_states = self.ln_f(op, hidden_states)
        return hidden_states, present_key_values


class FalconCausalLMModel(nn.Module):
    """Falcon/Bloom CausalLM with ALiBi or RoPE positional encoding.

    Supports both ALiBi (Falcon-40B, 180B, Bloom) and RoPE (Falcon-7B)
    positional encoding, as well as parallel and sequential attention+MLP.

    Outer attribute ``transformer`` matches HF Falcon weight prefix.

    Replicates HuggingFace's ``FalconForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.transformer = _FalconTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Falcon/Bloom weight names to our names.

        Most names now match HF directly after attribute alignment.
        Only output projection rename (dense→o_proj) and fused QKV
        splitting remain.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle fused QKV
            if "query_key_value" in key:
                q, k, v = split_fused_qkv(
                    value,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                base = key.replace(
                    "self_attention.query_key_value.",
                    "self_attention.",
                )
                new_state_dict[base.replace("self_attention.", "self_attention.q_proj.")] = q
                new_state_dict[base.replace("self_attention.", "self_attention.k_proj.")] = k
                new_state_dict[base.replace("self_attention.", "self_attention.v_proj.")] = v
                continue

            # Output proj: HF uses "dense", Attention component uses "o_proj"
            new_key = key.replace("self_attention.dense.", "self_attention.o_proj.")
            new_state_dict[new_key] = value

        # Handle weight tying
        if self.config.tie_word_embeddings:
            embed_key = "transformer.word_embeddings.weight"
            head_key = "lm_head.weight"
            if head_key not in new_state_dict and embed_key in new_state_dict:
                new_state_dict[head_key] = new_state_dict[embed_key]

        return new_state_dict
