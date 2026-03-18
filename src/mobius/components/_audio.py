# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio encoder components (Conformer-style).

Provides modules for Conformer audio encoding in multimodal models:
- ``NeMoConvSubsampling``: Conv2d-based audio subsampling (NeMo-style)
- ``MeanVarianceNorm``: Input feature normalization
- ``T5RelativeAttentionBias``: Relative position bias for attention
- ``ConformerFeedForward``: Macaron-style feed-forward with GLU
- ``ConformerConvModule``: Conformer convolution module
- ``ConformerAttention``: Multi-head attention with relative bias
- ``ConformerEncoderLayer``: Full conformer block
- ``ConformerEncoder``: Stack of conformer layers with subsampling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Embedding, LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


def _swish(op: builder.OpBuilder, x):
    """Swish activation: x * sigmoid(x)."""
    return op.Mul(x, op.Sigmoid(x))


# ---------------------------------------------------------------------------
# Private helper modules
# ---------------------------------------------------------------------------


class _SwishModule(nn.Module):
    """Swish activation as an nn.Module (no parameters)."""

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Mul(x, op.Sigmoid(x))


class _IdentityModule(nn.Module):
    """Identity (no-op) module, placeholder for dropout layers."""

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return x


class _Conv2d(nn.Module):
    """Conv2d with stored convolution parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            [out_channels, in_channels // groups, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter([out_channels])
        self._kernel_shape = [kernel_size, kernel_size]
        self._strides = [stride, stride]
        self._pads = [padding, padding, padding, padding]
        self._groups = groups

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
            pads=self._pads,
            group=self._groups,
        )


class _ConvWeight(nn.Module):
    """Conv1d wrapper: stores weight/bias and exposes forward()."""

    def __init__(
        self,
        weight_shape: list[int],
        *,
        kernel_shape: list[int] | None = None,
        strides: list[int] | None = None,
        pads: list[int] | None = None,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight_shape)
        self.bias = nn.Parameter([weight_shape[0]])
        self._kernel_shape = kernel_shape or [weight_shape[2]]
        self._strides = strides or [1]
        self._pads = pads or [0, 0]
        self._groups = groups

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
            pads=self._pads,
            group=self._groups,
        )


class _GLULinear(nn.Module):
    """GLU linear layer: Linear(in, out*2) → split → swish gate."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = Linear(input_dim, output_dim * 2)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.linear(op, x)  # [B, T, output_dim * 2]
        first, second = op.Split(x, axis=-1, num_outputs=2, _outputs=2)
        return op.Mul(first, _swish(op, second))


class _GLUPointWiseConv(nn.Module):
    """GLU pointwise convolution for ConformerConvModule.

    Contains a pointwise Conv1d that doubles channels, plus two bias
    parameters for the GLU gate.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ext_pw_conv_1d = _ConvWeight([channels * 2, channels, 1])
        self.b1 = nn.Parameter([1, channels, 1])
        self.b2 = nn.Parameter([1, channels, 1])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # [B, C, T] → [B, 2C, T] → split → GLU gate
        glu_out = self.ext_pw_conv_1d(op, x)  # [B, 2C, T]
        first, second = op.Split(glu_out, axis=1, num_outputs=2, _outputs=2)
        first = op.Add(first, self.b1)
        second = op.Add(second, self.b2)
        return op.Mul(first, _swish(op, second))


class _DepthwiseSepConv(nn.Module):
    """Depthwise separable convolution: depthwise Conv1d + pointwise Conv1d."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.dw_conv = _ConvWeight(
            [channels, 1, kernel_size],
            kernel_shape=[kernel_size],
            pads=[kernel_size // 2, kernel_size // 2],
            groups=channels,
        )
        self.pw_conv = _ConvWeight([channels, channels, 1])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.dw_conv(op, x)
        return self.pw_conv(op, x)


# ---------------------------------------------------------------------------
# Public components
# ---------------------------------------------------------------------------


class NeMoConvSubsampling(nn.Module):
    """NeMo-style Conv2d subsampling for audio features.

    Three stride-2 convolution stages reduce the time dimension by 8x.
    A depthwise-separable pattern is used for the 2nd and 3rd stages.

    Input:  ``[batch, time, input_size]``
    Output: ``[batch, time // 8, feat_out]``
    """

    def __init__(self, input_size: int, conv_channels: int, feat_out: int):
        super().__init__()

        # Compute output frequency dimension after 3 stride-2 convolutions
        freq = input_size
        for _ in range(3):
            freq = (freq + 2 - 3) // 2 + 1

        c = conv_channels
        self.conv = nn.ModuleList(
            [
                _Conv2d(1, c, kernel_size=3, stride=2, padding=1),  # 0
                _SwishModule(),  # 1
                _Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c),  # 2
                _Conv2d(c, c, kernel_size=1),  # 3
                _SwishModule(),  # 4
                _Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c),  # 5
                _Conv2d(c, c, kernel_size=1),  # 6
                _SwishModule(),  # 7
            ]
        )
        self.out = Linear(conv_channels * freq, feat_out)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, input_size]
        x = op.Unsqueeze(x, [1])  # [B, 1, T, F]

        for layer in self.conv:
            x = layer(op, x)

        # Reshape: [B, C, T', F'] → [B, T', C * F']
        x = op.Transpose(x, perm=[0, 2, 1, 3])
        x = op.Reshape(x, [0, 0, -1])

        # Linear projection to feat_out
        return self.out(op, x)


class MeanVarianceNorm(nn.Module):
    """Mean-variance normalization for audio features.

    Applies ``(x - mean) * invstd`` using learned global statistics.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.global_mean = nn.Parameter([input_size])
        self.global_invstd = nn.Parameter([input_size])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = op.Sub(x, self.global_mean)
        return op.Mul(x, self.global_invstd)


class T5RelativeAttentionBias(nn.Module):
    """T5-style relative position bias for attention.

    Computes pairwise relative position indices, shifts by max_distance,
    and gathers learned bias values per head.

    Output: ``[1, num_heads, seq_len, seq_len]``
    """

    def __init__(self, num_buckets: int, num_heads: int, max_distance: int):
        super().__init__()
        self.bias_values = Embedding(num_buckets, num_heads)
        self._max_distance = max_distance
        self._num_buckets = num_buckets

    def forward(self, op: builder.OpBuilder, seq_length: ir.Value):
        # seq_length: scalar tensor (int64)
        zero = op.Constant(value_int=0)
        one = op.Constant(value_int=1)
        positions = op.Range(zero, seq_length, one)  # [seq]

        # Pairwise relative positions: query_pos - key_pos
        q_pos = op.Unsqueeze(positions, [1])  # [seq, 1]
        k_pos = op.Unsqueeze(positions, [0])  # [1, seq]
        relative_pos = op.Sub(q_pos, k_pos)  # [seq, seq]

        # Shift and clip to valid bucket range
        shifted = op.Add(relative_pos, op.Constant(value_int=self._max_distance))
        clipped = op.Clip(
            shifted,
            op.Constant(value_int=0),
            op.Constant(value_int=self._num_buckets - 1),
        )

        # Gather bias values: [seq, seq] → [seq, seq, num_heads]
        bias = self.bias_values(op, clipped)

        # Rearrange to [1, num_heads, seq, seq]
        bias = op.Transpose(bias, perm=[2, 0, 1])
        return op.Unsqueeze(bias, [0])


class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward module with GLU activation.

    Structure: LayerNorm → GLULinear → Linear

    GLULinear splits its output in half and applies a gated activation:
    ``first_half * swish(second_half)``.
    """

    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.net = nn.ModuleList(
            [
                _GLULinear(d_model, d_inner),  # 0: GLU (linear + split + gate)
                _IdentityModule(),  # 1: dropout (no-op in inference)
                Linear(d_inner, d_model),  # 2: projection
            ]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.layer_norm(op, x)
        for layer in self.net:
            x = layer(op, x)
        return x


class ConformerConvModule(nn.Module):
    """Conformer convolution module.

    Structure (all Conv1d in ``[B, C, T]`` layout internally):
        LayerNorm → GLU-PointWise-Conv → Depthwise-Sep-Conv → Swish → PointWise-Conv
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self._channels = channels
        self._kernel_size = kernel_size
        self.layer_norm = LayerNorm(channels)
        self.glu = _GLUPointWiseConv(channels)
        self.dw_sep_conv_1d = _DepthwiseSepConv(channels, kernel_size)
        self.ext_pw_conv_1d = _ConvWeight([channels, channels, 1])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, C]
        x = self.layer_norm(op, x)
        x = op.Transpose(x, perm=[0, 2, 1])  # [B, C, T]

        # GLU pointwise conv: [B, C, T] → gated [B, C, T]
        x = self.glu(op, x)

        # Depthwise separable conv: [B, C, T] → [B, C, T]
        x = self.dw_sep_conv_1d(op, x)

        # Swish + final pointwise conv
        x = _swish(op, x)
        x = self.ext_pw_conv_1d(op, x)

        return op.Transpose(x, perm=[0, 2, 1])  # [B, T, C]


class ConformerAttention(nn.Module):
    """Multi-head attention with relative position bias.

    Uses the ONNX Attention op (opset 23) with separate Q/K/V/O projections.
    The T5 relative bias is passed as ``attn_mask`` (attention bias) to the
    Attention op.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._scale = float(self._head_dim) ** -0.5
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)

    def forward(self, op: builder.OpBuilder, x: ir.Value, relative_attention_bias: ir.Value):
        # x: [B, T, D], relative_attention_bias: [1, H, T, T]
        q = self.linear_q(op, x)
        k = self.linear_k(op, x)
        v = self.linear_v(op, x)

        # op.Attention expects [B, T, H*D_h] for Q/K/V and [1, H, T, T] for bias
        attn_output = op.Attention(
            q,
            k,
            v,
            relative_attention_bias,
            kv_num_heads=self._num_heads,
            q_num_heads=self._num_heads,
            scale=self._scale,
            _outputs=1,
        )

        return self.linear_out(op, attn_output)


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer (Macaron structure).

    Forward::

        x += 0.5 * feed_forward_in(x)
        x += self_attn(layer_norm_att(x), relative_bias)
        x += conv(x)
        x += 0.5 * feed_forward_out(x)
        x = layer_norm(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_inner: int,
        kernel_size: int,
    ):
        super().__init__()
        self.feed_forward_in = ConformerFeedForward(d_model, d_inner)
        self.self_attn = ConformerAttention(d_model, num_heads)
        self.conv = ConformerConvModule(d_model, kernel_size)
        self.feed_forward_out = ConformerFeedForward(d_model, d_inner)
        self.layer_norm_att = LayerNorm(d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, op: builder.OpBuilder, x: ir.Value, relative_attention_bias: ir.Value):
        half = op.Constant(value_float=0.5)

        # Macaron feed-forward in
        x = op.Add(x, op.Mul(self.feed_forward_in(op, x), half))

        # Multi-head attention with pre-norm
        norm_x = self.layer_norm_att(op, x)
        x = op.Add(x, self.self_attn(op, norm_x, relative_attention_bias))

        # Convolution module
        x = op.Add(x, self.conv(op, x))

        # Macaron feed-forward out
        x = op.Add(x, op.Mul(self.feed_forward_out(op, x), half))

        return self.layer_norm(op, x)


class ConformerEncoder(nn.Module):
    """Conformer audio encoder.

    Combines mean-variance normalization, conv subsampling, relative
    attention bias, and a stack of Conformer encoder layers.

    Input:  ``[batch, time, input_size]``  (e.g., 80-dim mel spectrogram)
    Output: ``[batch, time // 8, attention_dim]``
    """

    def __init__(
        self,
        input_size: int = 80,
        attention_dim: int = 1024,
        attention_heads: int = 16,
        num_blocks: int = 24,
        linear_units: int = 1536,
        kernel_size: int = 3,
        conv_channels: int = 1024,
        t5_bias_max_distance: int = 500,
    ):
        super().__init__()
        num_buckets = t5_bias_max_distance * 2

        self.embed = NeMoConvSubsampling(input_size, conv_channels, attention_dim)
        self.encoder_embedding = MeanVarianceNorm(input_size)
        self.relative_attention_bias_layer = T5RelativeAttentionBias(
            num_buckets,
            attention_heads,
            t5_bias_max_distance,
        )

        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    attention_dim,
                    attention_heads,
                    linear_units,
                    kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, op: builder.OpBuilder, audio_features: ir.Value):
        # audio_features: [B, T, input_size]
        x = self.encoder_embedding(op, audio_features)
        x = self.embed(op, x)  # [B, T', attention_dim]

        # Compute relative attention bias from subsampled sequence length
        seq_length = op.Shape(x, start=1, end=2)
        rel_bias = self.relative_attention_bias_layer(op, seq_length)

        for layer in self.encoders:
            x = layer(op, x, rel_bias)

        return x
