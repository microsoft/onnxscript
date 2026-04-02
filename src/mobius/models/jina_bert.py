# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Jina BERT encoder model with ALiBi position encoding and GEGLU MLP.

Jina BERT (jinaai/jina-embeddings-v2-base-en) is a BERT variant that
replaces absolute position embeddings with Attention with Linear Biases
(ALiBi) and uses a Gated Linear Unit (GEGLU) feed-forward network.

Key differences from standard BERT:

    Embeddings:
        word_embeddings + token_type_embeddings + LayerNorm
        (no position_embeddings — ALiBi replaces them)

Attention:
        Same Q/K/V projections as BERT, but adds ALiBi bias:
        ``attn_score[h, i, j] += -slope_h * |i - j|``
        Slopes are a geometric sequence per head.

    Feed-forward:
        GEGLU instead of standard MLP:
        ``gated, non_gated = Linear(x, 2 * intermediate_size).split(2)``
        ``output = Linear(GELU(gated) * non_gated, hidden_size)``
        With post-norm: ``LayerNorm(output + residual)``

    HF weight naming:
        encoder.layer.N.attention.self.{query,key,value}.{weight,bias}
        encoder.layer.N.attention.output.dense.{weight,bias}
        encoder.layer.N.attention.output.LayerNorm.{weight,bias}
        encoder.layer.N.mlp.gated_layers.weight   (no bias)
        encoder.layer.N.mlp.wo.{weight,bias}
        encoder.layer.N.mlp.layernorm.{weight,bias}
        embeddings.word_embeddings.weight
        embeddings.token_type_embeddings.weight
        embeddings.LayerNorm.{weight,bias}

Replicates HuggingFace ``JinaBertModel`` from
``jinaai/jina-bert-implementation``.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Embedding, LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# ALiBi bias computation
# ---------------------------------------------------------------------------


def _compute_alibi_slopes(num_heads: int) -> list[float]:
    """Compute per-head ALiBi slopes as a geometric sequence.

    Following the ALiBi paper (Press et al., 2021), slopes are:
        2^(-8/n * k) for k in 1..n when n is a power of 2.
    When n is not a power of 2, use the interleaved workaround.
    """

    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return _get_slopes_power_of_2(num_heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    slopes = _get_slopes_power_of_2(closest_power_of_2)
    extra = _get_slopes_power_of_2(2 * closest_power_of_2)
    slopes += extra[0::2][: num_heads - closest_power_of_2]
    return slopes


# ---------------------------------------------------------------------------
# Jina BERT components
# ---------------------------------------------------------------------------


class _JinaBertEmbeddings(nn.Module):
    """Jina BERT embeddings: word + token_type + LayerNorm (no positions).

    ALiBi handles positional information in the attention layer,
    so there is no position_embeddings table.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        layer_norm_eps: float,
        pad_token_id: int,
    ):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, hidden_size, pad_token_id)
        self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, token_type_ids: ir.Value):
        word_embeds = self.word_embeddings(op, input_ids)
        token_type_embeds = self.token_type_embeddings(op, token_type_ids)
        return self.LayerNorm(op, op.Add(word_embeds, token_type_embeds))


class _JinaBertSelfAttention(nn.Module):
    """Self-attention projections: query, key, value (HF naming)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = Linear(hidden_size, hidden_size, bias=True)
        self.key = Linear(hidden_size, hidden_size, bias=True)
        self.value = Linear(hidden_size, hidden_size, bias=True)


class _JinaBertAttentionOutput(nn.Module):
    """Attention output: dense projection + post-norm (HF naming)."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=True)
        self.LayerNorm = LayerNorm(hidden_size, eps=eps)


class _JinaBertAttention(nn.Module):
    """Jina BERT attention with ALiBi bias.

    Produces HF-compatible paths:
        attention.self.query.weight
        attention.output.dense.weight
        attention.output.LayerNorm.weight
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float):
        super().__init__()
        self_attn = _JinaBertSelfAttention(hidden_size, num_heads)
        self.self = self_attn
        self.output = _JinaBertAttentionOutput(hidden_size, eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        alibi_bias: ir.Value,
    ):
        self_attn = self.self
        query = self_attn.query(op, hidden_states)
        key = self_attn.key(op, hidden_states)
        value = self_attn.value(op, hidden_states)

        # Combine attention_mask and ALiBi bias into a single additive mask
        # attention_mask: (B, 1, 1, S) additive mask (0 or -10000)
        # alibi_bias: (1, H, S, S) pre-computed ALiBi bias
        combined_mask = op.Add(attention_mask, alibi_bias)

        attn_out = op.Attention(
            query,
            key,
            value,
            combined_mask,
            q_num_heads=self_attn.num_heads,
            kv_num_heads=self_attn.num_heads,
            scale=float(self_attn.head_dim**-0.5),
        )

        # Post-norm: dense + residual + LayerNorm
        attn_out = self.output.dense(op, attn_out)
        return self.output.LayerNorm(op, op.Add(hidden_states, attn_out))


class _JinaBertGLUMLP(nn.Module):
    """GEGLU feed-forward with residual connection and post-LayerNorm.

    Architecture:
        gated, non_gated = Linear(x, 2 * intermediate_size).split(2)
        hidden = GELU(gated) * non_gated
        output = LayerNorm(Linear(hidden, hidden_size) + residual)

    HF weight paths:
        mlp.gated_layers.weight  (no bias)
        mlp.wo.weight / mlp.wo.bias
        mlp.layernorm.weight / mlp.layernorm.bias
    """

    def __init__(self, hidden_size: int, intermediate_size: int, layer_norm_eps: float):
        super().__init__()
        self._intermediate_size = intermediate_size
        # Projects to 2x intermediate: first half is gated, second is value
        self.gated_layers = Linear(hidden_size, intermediate_size * 2, bias=False)
        self.wo = Linear(intermediate_size, hidden_size, bias=True)
        self.layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states

        # Project to 2x intermediate and split
        projected = self.gated_layers(op, hidden_states)
        # Split along last axis into gate and up halves
        half = self._intermediate_size
        gated = op.Slice(
            projected,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[-1]),
        )
        non_gated = op.Slice(
            projected,
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[half * 2]),
            op.Constant(value_ints=[-1]),
        )

        # GEGLU: GELU(gated) * non_gated
        hidden = op.Mul(op.Gelu(gated), non_gated)

        # Down-project + residual + post-LayerNorm
        output = self.wo(op, hidden)
        return self.layernorm(op, op.Add(output, residual))


class _JinaBertEncoderLayer(nn.Module):
    """Jina BERT encoder layer: ALiBi attention + GEGLU MLP.

    HF weight paths:
        layer.N.attention.self.{query,key,value}.{weight,bias}
        layer.N.attention.output.{dense,LayerNorm}.{weight,bias}
        layer.N.mlp.{gated_layers,wo,layernorm}.{weight,bias}
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attention = _JinaBertAttention(hidden_size, num_attention_heads, layer_norm_eps)
        self.mlp = _JinaBertGLUMLP(hidden_size, intermediate_size, layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        alibi_bias: ir.Value,
    ):
        hidden_states = self.attention(op, hidden_states, attention_mask, alibi_bias)
        hidden_states = self.mlp(op, hidden_states)
        return hidden_states


class _JinaBertEncoder(nn.Module):
    """Jina BERT encoder: stack of layers with shared ALiBi bias.

    Computes ALiBi bias once and passes to all layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._num_heads = config.num_attention_heads
        self.layer = nn.ModuleList(
            [
                _JinaBertEncoderLayer(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    layer_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
    ):
        # Build ALiBi bias dynamically from sequence length
        alibi_bias = self._build_alibi_bias(op, hidden_states)

        for layer in self.layer:
            hidden_states = layer(op, hidden_states, attention_mask, alibi_bias)
        return hidden_states

    def _build_alibi_bias(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Compute bidirectional ALiBi bias: -slope_h * |i - j|.

        Args:
            hidden_states: (B, S, D) — used to derive sequence length.

        Returns:
            alibi_bias: (1, num_heads, S, S) additive attention bias.
        """
        num_heads = self._num_heads

        # Pre-compute slopes as a constant: [num_heads]
        slopes = _compute_alibi_slopes(num_heads)
        slopes_np = np.array(slopes, dtype=np.float32)

        # seq_len from hidden_states shape
        seq_len = op.Squeeze(op.Shape(hidden_states, start=1, end=2))

        # Position indices: [0, 1, ..., seq_len-1]
        positions = op.Cast(
            op.Range(
                op.Constant(value_int=0),
                seq_len,
                op.Constant(value_int=1),
            ),
            to=1,  # FLOAT
        )

        # Distance matrix: |i - j| for all pairs
        # positions_row: (S, 1), positions_col: (1, S)
        pos_row = op.Unsqueeze(positions, [1])  # (S, 1)
        pos_col = op.Unsqueeze(positions, [0])  # (1, S)
        distance = op.Abs(op.Sub(pos_row, pos_col))  # (S, S)

        # Negate: -|i - j|
        neg_distance = op.Neg(distance)  # (S, S)

        # Apply slopes: slopes[h] * -|i - j| → (num_heads, S, S)
        # slopes: (num_heads, 1, 1) * neg_distance: (1, S, S) → (H, S, S)
        slopes_const = op.Constant(value_floats=slopes_np.tolist())
        slopes_3d = op.Unsqueeze(slopes_const, [1, 2])  # (H, 1, 1)
        neg_dist_3d = op.Unsqueeze(neg_distance, [0])  # (1, S, S)
        alibi = op.Mul(slopes_3d, neg_dist_3d)  # (H, S, S)

        # Add batch dim: (1, H, S, S)
        return op.Unsqueeze(alibi, [0])


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class JinaBertModel(nn.Module):
    """Jina BERT encoder with ALiBi + GEGLU for feature extraction.

    Replicates ``JinaBertModel`` from jinaai/jina-bert-implementation.

    Args:
        config: ArchitectureConfig with BERT-like fields.
    """

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embeddings = _JinaBertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=getattr(config, "type_vocab_size", 2),
            layer_norm_eps=config.rms_norm_eps,
            pad_token_id=config.pad_token_id or 0,
        )
        self.encoder = _JinaBertEncoder(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,
    ):
        hidden_states = self.embeddings(op, input_ids, token_type_ids)

        # Convert attention_mask from (B, S) int to (B, 1, 1, S) additive
        # 1 → 0.0 (attend), 0 → -10000.0 (mask)
        mask_float = op.Cast(attention_mask, to=1)  # FLOAT
        mask_inv = op.Sub(op.Constant(value_float=1.0), mask_float)
        additive_mask = op.Mul(mask_inv, op.Constant(value_float=-10000.0))
        additive_mask = op.Unsqueeze(additive_mask, [1, 2])  # (B, 1, 1, S)

        hidden_states = self.encoder(op, hidden_states, additive_mask)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF Jina BERT weight names to match our parameter names.

        Strips the ``bert.`` prefix and handles gamma/beta compat.
        Skips pooler, position_embeddings, and ALiBi buffer.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_jina_bert_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------

# Old checkpoints use gamma/beta instead of weight/bias
_PARAM_RENAMES = {"gamma": "weight", "beta": "bias"}

# Prefixes to skip entirely
_SKIP_PREFIXES = (
    "cls.",  # Classification head (not used for embeddings)
    "bert.pooler.",  # Pooler layer
)

# Exact names to skip
_SKIP_NAMES = frozenset(
    {
        "position_ids",
        "alibi",
    }
)


def _rename_jina_bert_weight(name: str) -> str | None:
    """Map a single HF weight name to our module parameter name."""
    # Skip classification heads, pooler, buffers
    if any(name.startswith(p) for p in _SKIP_PREFIXES):
        return None
    basename = name.rsplit(".", 1)[-1] if "." in name else name
    if basename in _SKIP_NAMES or name in _SKIP_NAMES:
        return None
    # Skip position_embeddings (ALiBi has none)
    if "position_embeddings" in name:
        return None

    # Strip model prefix: bert. or jina_bert.
    new_name = re.sub(r"^(bert|jina_bert)\.", "", name)

    # Rename gamma/beta → weight/bias
    parts = new_name.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in _PARAM_RENAMES:
        new_name = f"{parts[0]}.{_PARAM_RENAMES[parts[1]]}"

    return new_name
