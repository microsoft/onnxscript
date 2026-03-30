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

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv
from mobius.components import (
    FCMLP,
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
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="gelu",
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
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="gelu",
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
        dual_ln = config.dual_ln

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


class _BloomTextModel(_FalconTextModel):
    """Bloom text model with word_embeddings_layernorm after token embedding.

    HF Bloom applies LayerNorm to word embeddings before the transformer
    layers, which Falcon does not.  Attribute name matches the HF weight
    ``transformer.word_embeddings_layernorm.{weight,bias}``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.word_embeddings_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.word_embeddings(op, input_ids)
        # Bloom applies LayerNorm to word embeddings before transformer
        hidden_states = self.word_embeddings_layernorm(op, hidden_states)
        present_key_values = []

        if self._alibi:
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


def _split_falcon_interleaved_qkv(
    fused: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Falcon's interleaved fused QKV weight into separate Q, K, V.

    HF Falcon (``new_decoder_architecture=True``) groups the fused QKV
    weight as ``[Q0..Q_{g-1}, K, V]`` per KV-group, where
    ``g = num_heads // num_kv_heads``.  Each block is ``head_dim`` rows.

    Args:
        fused: Weight tensor, shape ``(total_qkv, hidden_size)`` for weights
               or ``(total_qkv,)`` for biases.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimension per head.

    Returns:
        (Q, K, V) tensors with shapes
        ``(num_heads * head_dim, ...)``,
        ``(num_kv_heads * head_dim, ...)``,
        ``(num_kv_heads * head_dim, ...)``.
    """
    group_size = num_heads // num_kv_heads  # Q heads per KV group
    # Reshape: (num_kv_heads, group_size + 2, head_dim, ...)
    shape = (num_kv_heads, group_size + 2, head_dim, *fused.shape[1:])
    grouped = fused.reshape(shape)

    q_parts = grouped[:, :group_size]  # (num_kv_heads, group_size, head_dim, ...)
    k_parts = grouped[:, group_size]  # (num_kv_heads, head_dim, ...)
    v_parts = grouped[:, group_size + 1]  # (num_kv_heads, head_dim, ...)

    # Flatten Q: (num_kv_heads, group_size, head_dim, ...) → (num_heads * head_dim, ...)
    q = q_parts.reshape(num_heads * head_dim, *fused.shape[1:])
    k = k_parts.reshape(num_kv_heads * head_dim, *fused.shape[1:])
    v = v_parts.reshape(num_kv_heads * head_dim, *fused.shape[1:])
    return q, k, v


class FalconCausalLMModel(nn.Module):
    """Falcon CausalLM with ALiBi or RoPE positional encoding.

    Supports both ALiBi (Falcon-40B, 180B) and RoPE (Falcon-7B)
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

        Falcon with ``new_decoder_architecture=True`` uses an interleaved
        QKV layout: for each KV-group the weight rows are
        ``[Q_0, Q_1, ..., Q_{g-1}, K, V]`` where ``g = num_heads / num_kv_heads``.
        This differs from the standard contiguous ``[Q..., K..., V...]`` layout.
        """
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle fused QKV — Falcon interleaved layout
            if "query_key_value" in key:
                q, k, v = _split_falcon_interleaved_qkv(
                    value,
                    num_heads,
                    num_kv_heads,
                    head_dim,
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
            # MLP: HF uses dense_h_to_4h/dense_4h_to_h, FCMLP uses up_proj/down_proj
            new_key = new_key.replace(".mlp.dense_h_to_4h.", ".mlp.up_proj.")
            new_key = new_key.replace(".mlp.dense_4h_to_h.", ".mlp.down_proj.")
            new_state_dict[new_key] = value

        # Handle weight tying
        if self.config.tie_word_embeddings:
            embed_key = "transformer.word_embeddings.weight"
            head_key = "lm_head.weight"
            if head_key not in new_state_dict and embed_key in new_state_dict:
                new_state_dict[head_key] = new_state_dict[embed_key]

        return new_state_dict


class BloomCausalLMModel(FalconCausalLMModel):
    """Bloom CausalLM with ALiBi and word_embeddings_layernorm.

    Bloom applies LayerNorm to word embeddings before the transformer
    blocks, which standard Falcon does not.

    Replicates HuggingFace's ``BloomForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        # Bloom always uses ALiBi positional encoding — enforce it regardless
        # of whether the caller set alibi=True in the config, since HF's
        # BloomConfig has no alibi field and ArchitectureConfig defaults to False.
        config = dataclasses.replace(config, alibi=True)
        super().__init__(config)
        self.transformer = _BloomTextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Bloom weight names to our names.

        Bloom uses ``input_layernorm`` / ``post_attention_layernorm``
        while our ``_ALiBiDecoderLayer`` uses ``ln_attn`` / ``ln_mlp``.
        """
        renamed = {}
        for key, value in state_dict.items():
            new_key = key.replace(".input_layernorm.", ".ln_attn.")
            new_key = new_key.replace(".post_attention_layernorm.", ".ln_mlp.")
            renamed[new_key] = value
        return super().preprocess_weights(renamed)


class MPTCausalLMModel(FalconCausalLMModel):
    """MPT (MosaicML Pretrain Transformer) causal language model.

    MPT uses a sequential pre-norm architecture with ALiBi positional encoding:

        norm_1 → attention → residual → norm_2 → MLP → residual

    This is the standard pre-norm transformer (not parallel attention), with
    two separate LayerNorms per block.

    Weight naming differences from Falcon:

    - ``transformer.blocks.N.*`` instead of ``transformer.h.N.*``
    - ``norm_1`` / ``norm_2`` instead of ``ln_attn`` / ``ln_mlp``
    - ``attn.Wqkv`` (fused QKV) instead of ``self_attention.query_key_value``
    - ``attn.out_proj`` instead of ``self_attention.dense``
    - ``ffn.up_proj`` / ``ffn.down_proj`` (already matches our naming!)
    - ``transformer.wte`` instead of ``transformer.word_embeddings``
    - ``transformer.norm_f`` instead of ``transformer.ln_f``
    - Uses ALiBi positional encoding (no RoPE)

    MPT only supports MHA (no GQA), so ``num_key_value_heads`` is forced to
    match ``num_attention_heads``.

    Replicates HuggingFace's ``MptForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        # MPT is MHA only — override KV heads to prevent shape mismatches
        config = dataclasses.replace(config, num_key_value_heads=config.num_attention_heads)
        # MPT uses ALiBi (no RoPE) and sequential pre-norm (standard transformer):
        # norm_1 → attn → residual → norm_2 → mlp → residual
        # parallel_attn=False enables sequential computation with separate LNs.
        config = dataclasses.replace(config, alibi=True, parallel_attn=False)
        super().__init__(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF MPT weight names to FalconCausalLMModel attribute names.

        MPT uses a different naming convention from Falcon. This method
        translates all MPT-specific names into Falcon-compatible ones before
        delegating to ``FalconCausalLMModel.preprocess_weights``:

        1. Blocks prefix: ``transformer.blocks.N.`` → ``transformer.h.N.``
        2. Norms: ``norm_1.`` → ``ln_attn.``, ``norm_2.`` → ``ln_mlp.``
        3. Fused QKV: ``attn.Wqkv.`` → split into ``q_proj``, ``k_proj``, ``v_proj``
           (MPT uses contiguous ``[Q, K, V]`` layout, unlike Falcon's interleaved)
        4. Output proj: ``attn.out_proj.`` → ``self_attention.dense.``
        5. FFN: ``ffn.up_proj.`` / ``ffn.down_proj.`` → ``mlp.up_proj.`` / ``mlp.down_proj.``
           (FalconCausalLMModel.preprocess_weights converts ``dense_h_to_4h``/
           ``dense_4h_to_h`` to up/down, but MPT already uses up/down naming)
        6. Embeddings: ``transformer.wte.`` → ``transformer.word_embeddings.``
        7. Final norm: ``transformer.norm_f.`` → ``transformer.ln_f.``
        """
        import re

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Block index rename: transformer.blocks.N.* → transformer.h.N.*
            key = re.sub(r"transformer\.blocks\.(\d+)\.", r"transformer.h.\1.", key)
            # Layer norm renames
            key = key.replace(".norm_1.", ".ln_attn.")
            key = key.replace(".norm_2.", ".ln_mlp.")
            # Attention: split contiguous QKV before renaming to Falcon format.
            # MPT uses [Q, K, V] contiguous layout (unlike Falcon's interleaved).
            if ".attn.Wqkv." in key:
                q, k, v = split_fused_qkv(
                    value,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                base = key.replace(".attn.Wqkv.", ".self_attention.")
                renamed[base.replace(".self_attention.", ".self_attention.q_proj.")] = q
                renamed[base.replace(".self_attention.", ".self_attention.k_proj.")] = k
                renamed[base.replace(".self_attention.", ".self_attention.v_proj.")] = v
                continue
            key = key.replace(".attn.out_proj.", ".self_attention.dense.")
            # FFN: MPT already uses up_proj/down_proj so map to Falcon's intermediate names
            # that FalconCausalLMModel.preprocess_weights will then convert to up/down
            key = key.replace(".ffn.up_proj.", ".mlp.dense_h_to_4h.")
            key = key.replace(".ffn.down_proj.", ".mlp.dense_4h_to_h.")
            # Top-level renames
            key = key.replace("transformer.wte.", "transformer.word_embeddings.")
            key = key.replace("transformer.norm_f.", "transformer.ln_f.")
            renamed[key] = value

        # MPT has no LayerNorm bias (bias=None), but our FalconDecoderLayer uses
        # LayerNorm which always allocates a bias parameter.  Initialize all LN
        # biases to zero so the computation matches HF MPT exactly.
        h = self.config.hidden_size
        for i in range(self.config.num_hidden_layers):
            renamed[f"transformer.h.{i}.ln_attn.bias"] = torch.zeros(h)
            renamed[f"transformer.h.{i}.ln_mlp.bias"] = torch.zeros(h)
        renamed["transformer.ln_f.bias"] = torch.zeros(h)

        # Delegate to FalconCausalLMModel to handle QKV splitting and final renames
        return super().preprocess_weights(renamed)
