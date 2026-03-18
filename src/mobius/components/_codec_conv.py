# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convolution and activation components for the Qwen3-TTS codec tokenizer.

Implements causal convolutions, transposed convolutions, ConvNeXt blocks,
SnakeBeta activation, and the decoder upsampling/residual blocks used by
``Qwen3TTSTokenizerV2Decoder``.

All causal convolutions use explicit left-padding so that each output
sample depends only on current and past inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Utility: Causal Convolutions
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """1-D causal convolution with left-padding.

    Pads the input on the left by ``(kernel_size - 1) * dilation`` so the
    output length equals the input length and the convolution is causal.

    HF class wraps ``nn.Conv1d`` as ``self.conv``, so parameter names
    are ``<name>.conv.weight`` and ``<name>.conv.bias``.

    Parameters:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor (default 1).
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel_size = kernel_size
        self._dilation = dilation
        # Named 'conv' to match HF MimiConv1d.conv weight names
        self.conv = _Conv1dParams(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            dilation=dilation,
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Forward pass.

        Args:
            x: (B, C_in, T) input tensor.

        Returns:
            (B, C_out, T) — same temporal length as input.
        """
        pad_left = (self._kernel_size - 1) * self._dilation
        if pad_left > 0:
            pads = [0, 0, pad_left, 0, 0, 0]
            x = op.Pad(x, op.Constant(value_ints=pads), mode="constant")

        return self.conv(op, x)


class CausalTransConv1d(nn.Module):
    """1-D causal transposed convolution (upsampling).

    Uses ``ConvTranspose`` then trims ``(kernel_size - stride)`` samples
    from the right to maintain causality.

    HF weight names: ``<name>.conv.weight``, ``<name>.conv.bias``.

    Parameters:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Upsampling factor.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._trim_right = kernel_size - stride
        # Named 'conv' to match HF MimiConvTranspose1d.conv weight names
        self.conv = _TransConv1dParams(in_channels, out_channels, kernel_size, stride, bias)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Upsample input.

        Args:
            x: (B, C_in, T).

        Returns:
            (B, C_out, T * stride).
        """
        y = self.conv(op, x)
        # Trim right to restore causality
        if self._trim_right > 0:
            y = op.Slice(
                y,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[-self._trim_right]),
                op.Constant(value_ints=[2]),  # axis=2 (time)
            )
        return y


# ---------------------------------------------------------------------------
# Parameter holders (match HF weight name structure)
# ---------------------------------------------------------------------------


class _Conv1dParams(nn.Module):
    """Holds Conv1d weight/bias and performs the convolution.

    Must be **called** (not just attribute-accessed) so onnxscript
    registers the parameters as graph initializers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        bias: bool = True,
        *,
        dilation: int = 1,
        stride: int = 1,
        group: int = 1,
    ):
        super().__init__()
        self._kernel = kernel
        self._dilation = dilation
        self._stride = stride
        self._group = group
        self.weight = nn.Parameter([out_ch, in_ch, kernel])
        self.bias = nn.Parameter([out_ch]) if bias else None

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        if self.bias is not None:
            return op.Conv(
                x,
                self.weight,
                self.bias,
                kernel_shape=[self._kernel],
                strides=[self._stride],
                dilations=[self._dilation],
                pads=[0, 0],
                group=self._group,
            )
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            dilations=[self._dilation],
            pads=[0, 0],
            group=self._group,
        )


class _TransConv1dParams(nn.Module):
    """Holds ConvTranspose1d weight/bias and performs the transposed conv.

    Must be **called** so onnxscript registers parameters.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel_size = kernel
        self._stride = stride
        # ConvTranspose weight shape: (in_ch, out_ch, kernel)
        self.weight = nn.Parameter([in_ch, out_ch, kernel])
        self.bias = nn.Parameter([out_ch]) if bias else None

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        if self.bias is not None:
            return op.ConvTranspose(
                x,
                self.weight,
                self.bias,
                kernel_shape=[self._kernel_size],
                strides=[self._stride],
            )
        return op.ConvTranspose(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
        )


# ---------------------------------------------------------------------------
# Activation: SnakeBeta
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """SnakeBeta activation: ``x + (1/b) * sin^2(a * x)``.

    ``a`` and ``b`` are per-channel learnable parameters (stored as log
    values and exponentiated at forward time).

    HF class: ``SnakeBeta`` in qwen_tts.

    Parameters:
        channels: Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter([channels])
        self.beta = nn.Parameter([channels])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Apply SnakeBeta activation.

        Args:
            x: (B, C, T) input.

        Returns:
            (B, C, T) activated output.
        """
        # Reshape params from (C,) to (1, C, 1) for broadcasting
        alpha = op.Exp(op.Unsqueeze(self.alpha, [0, 2]))
        beta = op.Exp(op.Unsqueeze(self.beta, [0, 2]))

        # sin^2(alpha * x)
        sin_val = op.Sin(op.Mul(x, alpha))
        sin_sq = op.Mul(sin_val, sin_val)

        # x + (1 / (beta + eps)) * sin^2(alpha * x)
        inv_beta = op.Reciprocal(op.Add(beta, 1e-9))
        return op.Add(x, op.Mul(inv_beta, sin_sq))


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Learnable per-channel scale factor.

    Applied element-wise: ``output = scale * input``.

    HF class: ``Qwen3TTSTokenizerV2DecoderLayerScale`` in qwen_tts.

    Parameters:
        dim: Number of channels/features.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter([dim])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Apply learned scale.

        Args:
            x: (..., dim) or (B, dim, T) tensor.

        Returns:
            Scaled tensor, same shape.
        """
        return op.Mul(x, self.scale)


# ---------------------------------------------------------------------------
# ConvNeXt Block
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv → norm → expand → GELU → contract.

    Used in the decoder's upsample stage.

    HF class: ``Qwen3TTSTokenizerV2ConvNeXtBlock`` in qwen_tts.

    Parameters:
        dim: Number of channels.
        kernel_size: Depthwise conv kernel (default 7).
        expansion: MLP expansion factor (default 4).
    """

    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self._kernel_size = kernel_size
        # Depthwise causal convolution (groups=dim)
        self.dwconv = _DepthwiseConv1dParams(dim, kernel_size)
        # LayerNorm (channels-last)
        self.norm = LayerNorm(dim)
        # Pointwise expansion/contraction
        self.pwconv1 = Linear(dim, dim * expansion, bias=True)
        self.pwconv2 = Linear(dim * expansion, dim, bias=True)
        # Gamma scale
        self.gamma = nn.Parameter([dim])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Forward pass with residual connection.

        Args:
            x: (B, C, T) channels-first input.

        Returns:
            (B, C, T) output.
        """
        residual = x

        # Causal depthwise conv: left-pad then call dwconv
        pad_left = self._kernel_size - 1
        h = op.Pad(
            x,
            op.Constant(value_ints=[0, 0, pad_left, 0, 0, 0]),
            mode="constant",
        )
        h = self.dwconv(op, h)

        # Channels-last for norm + MLP: (B, C, T) → (B, T, C)
        h = op.Transpose(h, perm=[0, 2, 1])
        h = self.norm(op, h)
        h = self.pwconv1(op, h)
        h = op.Gelu(h)
        h = self.pwconv2(op, h)

        # Gamma scale: (B, T, C) * (C,) → (B, T, C)
        h = op.Mul(h, self.gamma)

        # Back to channels-first: (B, T, C) → (B, C, T)
        h = op.Transpose(h, perm=[0, 2, 1])

        return op.Add(residual, h)


class _DepthwiseConv1dParams(nn.Module):
    """Depthwise Conv1d (groups=dim). Called, not just attribute-accessed."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self._dim = dim
        # Depthwise: weight shape (dim, 1, kernel_size), group=dim
        self.conv = _Conv1dParams(
            1,
            dim,
            kernel_size,
            bias=True,
            group=dim,
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Apply depthwise convolution (no padding — caller pads)."""
        return self.conv(op, x)


# ---------------------------------------------------------------------------
# Decoder Residual Unit
# ---------------------------------------------------------------------------


class DecoderResidualUnit(nn.Module):
    """Residual unit: SnakeBeta → CausalConv(k=7) → SnakeBeta → Conv1d(k=1).

    HF class: ``Qwen3TTSTokenizerV2DecoderDecoderResidualUnit`` in qwen_tts.

    Parameters:
        dim: Number of channels.
        dilation: Dilation for the first convolution.
    """

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Forward with residual.

        Args:
            x: (B, C, T).

        Returns:
            (B, C, T).
        """
        h = self.act1(op, x)
        h = self.conv1(op, h)
        h = self.act2(op, h)
        h = self.conv2(op, h)
        return op.Add(x, h)


# ---------------------------------------------------------------------------
# Decoder Block (upsample + residual units)
# ---------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    """Decoder block: SnakeBeta -> CausalTransConv -> 3x ResidualUnit.

    Each block halves the channel dimension and upsamples the time
    dimension by ``upsample_rate``.

    HF class: ``Qwen3TTSTokenizerV2DecoderDecoderBlock`` in qwen_tts.

    Parameters:
        in_dim: Input channels.
        out_dim: Output channels (typically in_dim // 2).
        upsample_rate: Temporal upsampling factor.
    """

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int):
        super().__init__()
        # block.0: SnakeBeta
        # block.1: CausalTransConv (upsample)
        # block.2,3,4: ResidualUnits with dilation 1, 3, 9
        self.block = nn.Sequential(
            SnakeBeta(in_dim),
            CausalTransConv1d(
                in_dim,
                out_dim,
                kernel_size=2 * upsample_rate,
                stride=upsample_rate,
            ),
            DecoderResidualUnit(out_dim, dilation=1),
            DecoderResidualUnit(out_dim, dilation=3),
            DecoderResidualUnit(out_dim, dilation=9),
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        """Upsample and refine.

        Args:
            x: (B, in_dim, T).

        Returns:
            (B, out_dim, T * upsample_rate).
        """
        return self.block(op, x)
