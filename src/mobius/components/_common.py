# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

# Used as Slice "end" to mean "all remaining elements along this axis".
INT64_MAX = 9223372036854775807


class Linear(nn.Module):
    """Linear (fully-connected) layer using ONNX ops."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        w_t = op.Transpose(self.weight, perm=[1, 0])
        result = op.MatMul(x, w_t)
        if self.bias is not None:
            result = op.Add(result, self.bias)
        return result


class Embedding(nn.Module):
    """Embedding layer using ONNX Gather op."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter([num_embeddings, embedding_dim])
        self.padding_idx = padding_idx

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        return op.Gather(self.weight, input_ids)


class LayerNorm(nn.Module):
    """Layer Normalization using ONNX LayerNormalization op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return op.LayerNormalization(
            hidden_states,
            self.weight,
            self.bias,
            epsilon=self.eps,
            axis=-1,
        )


class LayerNormNoAffine(nn.Module):
    """Layer Normalization without learnable affine parameters.

    Used in AdaLayerNorm where scale/shift come from a modulation projection,
    matching ``nn.LayerNorm(elementwise_affine=False)`` in PyTorch.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self._hidden_size = hidden_size
        self._eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # ONNX LayerNormalization requires a Scale input; use all-ones
        # since this is the no-affine variant (scale/shift come externally).
        # CastLike ensures Scale matches the input dtype (fp16/bf16/fp32).
        scale = op.Constant(value=ir.tensor(np.ones(self._hidden_size, dtype=np.float32)))
        scale = op.CastLike(scale, hidden_states)
        return op.LayerNormalization(hidden_states, scale, axis=-1, epsilon=self._eps)


class GroupNorm(nn.Module):
    """Group Normalization using ONNX GroupNormalization op."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((num_channels,))
        self.bias = nn.Parameter((num_channels,))
        self._num_groups = num_groups
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.GroupNormalization(
            x, self.weight, self.bias, num_groups=self._num_groups, epsilon=self._eps
        )


def create_attention_bias(
    op: builder.OpBuilder,
    input_ids,
    attention_mask,
    sliding_window: int | None = None,
    dtype: ir.DataType = ir.DataType.FLOAT,
):
    """Create causal attention bias for use in attention mechanisms.

    Args:
        op: The OpBuilder.
        input_ids: Input tensor of shape (batch_size, query_length).
        attention_mask: Attention mask of shape (batch_size, total_length).
        sliding_window: Optional sliding window size for local attention.
        dtype: Data type for the bias tensor. The masked value uses the
            minimum representable value for this dtype (e.g. -65504 for
            float16, -3.4e38 for float32).

    Returns:
        Attention bias tensor of shape (batch_size, 1, query_length, total_length).
    """
    # cumsum on attention_mask gives indices
    all_indices = op.CumSum(attention_mask, 1)  # axis=1

    # kv_indices: (batch_size, 1, total_length)
    kv_indices = op.Unsqueeze(all_indices, [1])

    # q_indices: take last query_length elements
    # We use Gather with negative indices via dynamic slicing
    # For simplicity, use the full indices and let the Attention op handle masking
    # Actually we need to implement this with shape ops

    # Get query_length and total_length from shapes
    query_length = op.Shape(input_ids, start=1, end=2)  # 1-D [1]
    total_length = op.Shape(attention_mask, start=1, end=2)  # 1-D [1]
    start = op.Sub(total_length, query_length)
    # q_indices_2d: (batch_size, query_length)
    q_indices_2d = op.Slice(all_indices, start, total_length, [1])
    # q_indices: (batch_size, query_length, 1)
    q_indices = op.Unsqueeze(q_indices_2d, [2])

    # Causal mask: q_indices >= kv_indices
    full_mask = op.GreaterOrEqual(q_indices, kv_indices)

    if sliding_window is not None:
        # Also mask out positions too far away
        dist = op.Sub(q_indices, kv_indices)
        within_window = op.Less(dist, sliding_window)
        full_mask = op.And(full_mask, within_window)

    # Combine with attention_mask
    attn_mask_bool = op.Cast(op.Unsqueeze(attention_mask, [1]), to=ir.DataType.BOOL)
    full_mask = op.And(attn_mask_bool, full_mask)

    # Convert to float bias: 0 where attended, dtype.min where masked
    mask_value = float(dtype.min)
    attention_bias = op.Where(full_mask, 0.0, mask_value)
    attention_bias = op.Cast(attention_bias, to=dtype)

    # Unsqueeze to (batch_size, 1, query_length, total_length)
    return op.Unsqueeze(attention_bias, [1])
