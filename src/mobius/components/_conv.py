# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convolution building blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

if TYPE_CHECKING:
    import onnx_ir as ir


class Conv2d(nn.Module):
    """2D convolution with bias.

    Matches ``torch.nn.Conv2d`` with ``bias=True``.  The default ``padding=0``
    follows PyTorch convention; callers should specify padding explicitly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._groups = groups

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        p = self._padding
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
            group=self._groups,
        )


class Conv2dNoBias(nn.Module):
    """2D convolution without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._groups = groups

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        p = self._padding
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
            group=self._groups,
        )


class BatchNorm2d(nn.Module):
    """2D batch normalization.

    Matches ``torch.nn.BatchNorm2d``.  Uses ONNX ``BatchNormalization`` op
    with frozen running statistics (inference mode).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter((num_features,))
        self.bias = nn.Parameter((num_features,))
        self.running_mean = nn.Parameter((num_features,))
        self.running_var = nn.Parameter((num_features,))
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.BatchNormalization(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            epsilon=self._eps,
        )


class ConvTranspose2d(nn.Module):
    """2D transposed convolution (deconvolution) with bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.weight = nn.Parameter((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        p = self._padding
        return op.ConvTranspose(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
        )
