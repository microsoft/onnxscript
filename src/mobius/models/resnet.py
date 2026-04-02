# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ResNet — residual convolutional network for image classification.

Implements the HuggingFace ``ResNetModel`` architecture with both
basic (2-layer) and bottleneck (3-layer) residual blocks.

Architecture overview::

    pixel_values (B, 3, H, W)
      → Stem: Conv2d 7x7/2 + BN + ReLU + MaxPool 3x3/2
      → Stage 1: N₁ x ResidualBlock(hidden_sizes[0])
      → Stage 2: N₂ x ResidualBlock(hidden_sizes[1], stride=2)
      → Stage 3: N₃ x ResidualBlock(hidden_sizes[2], stride=2)
      → Stage 4: N₄ x ResidualBlock(hidden_sizes[3], stride=2)
      → AdaptiveAvgPool2d → (B, hidden_sizes[-1], 1, 1)

Replicates HuggingFace ``ResNetModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ResNetConfig
from mobius.components._conv import BatchNorm2d, Conv2dNoBias

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (or identity activation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        activate: bool = True,
    ):
        super().__init__()
        self.convolution = Conv2dNoBias(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.normalization = BatchNorm2d(out_channels)
        self._activate = activate

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.convolution(op, x)
        x = self.normalization(op, x)
        if self._activate:
            x = op.Relu(x)
        return x


class _Shortcut(nn.Module):
    """1x1 convolution shortcut for dimension/stride matching."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.convolution = Conv2dNoBias(
            in_channels, out_channels, kernel_size=1, stride=stride
        )
        self.normalization = BatchNorm2d(out_channels)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.convolution(op, x)
        x = self.normalization(op, x)
        return x


class _BottleneckBlock(nn.Module):
    """Bottleneck residual block: 1x1 → 3x3 → 1x1 + shortcut.

    HuggingFace ``ResNetBottleNeckLayer`` structure::

        layer.0: Conv 1x1 (reduce)  + BN + ReLU
        layer.1: Conv 3x3 (spatial) + BN + ReLU
        layer.2: Conv 1x1 (expand)  + BN (no ReLU)
        shortcut: optional 1x1 conv + BN

    The stride-2 downsampling is applied in layer.1 (the 3x3 conv)
    when ``downsample_in_bottleneck=False`` (HF default for ResNet-50).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 4,
    ):
        super().__init__()
        mid_channels = out_channels // reduction
        # 1x1 reduce
        self.layer = nn.ModuleList(
            [
                _ConvBnRelu(in_channels, mid_channels, kernel_size=1),
                # 3x3 spatial (stride applied here)
                _ConvBnRelu(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                # 1x1 expand (no activation — applied after residual add)
                _ConvBnRelu(
                    mid_channels,
                    out_channels,
                    kernel_size=1,
                    activate=False,
                ),
            ]
        )
        # Shortcut when dimensions change
        self._use_shortcut = (in_channels != out_channels) or stride != 1
        if self._use_shortcut:
            self.shortcut = _Shortcut(in_channels, out_channels, stride)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        residual = x
        for conv_layer in self.layer:
            x = conv_layer(op, x)
        if self._use_shortcut:
            residual = self.shortcut(op, residual)
        return op.Relu(op.Add(x, residual))


class _BasicBlock(nn.Module):
    """Basic residual block: 3x3 → 3x3 + shortcut.

    HuggingFace ``ResNetBasicLayer`` structure::

        layer.0: Conv 3x3 + BN + ReLU
        layer.1: Conv 3x3 + BN (no ReLU)
        shortcut: optional 1x1 conv + BN
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _ConvBnRelu(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                _ConvBnRelu(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    activate=False,
                ),
            ]
        )
        self._use_shortcut = (in_channels != out_channels) or stride != 1
        if self._use_shortcut:
            self.shortcut = _Shortcut(in_channels, out_channels, stride)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        residual = x
        for conv_layer in self.layer:
            x = conv_layer(op, x)
        if self._use_shortcut:
            residual = self.shortcut(op, residual)
        return op.Relu(op.Add(x, residual))


# ---------------------------------------------------------------------------
# Encoder (stage stack)
# ---------------------------------------------------------------------------


class _ResNetStage(nn.Module):
    """One ResNet stage: sequence of residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        layer_type: str,
    ):
        super().__init__()
        block_cls = _BottleneckBlock if layer_type == "bottleneck" else _BasicBlock
        blocks = []
        for i in range(depth):
            s = stride if i == 0 else 1
            ic = in_channels if i == 0 else out_channels
            blocks.append(block_cls(ic, out_channels, stride=s))
        self.layers = nn.Sequential(*blocks)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.layers(op, x)


class _ResNetEncoder(nn.Module):
    """ResNet encoder: 4 stages of residual blocks."""

    def __init__(self, config: ResNetConfig):
        super().__init__()
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        layer_type = config.layer_type
        embedding_size = config.embedding_size

        # First stage takes from embedding_size, subsequent from previous
        in_channels = [embedding_size, *list(hidden_sizes[:-1])]
        stages = []
        for i, (ic, oc, d) in enumerate(zip(in_channels, hidden_sizes, depths)):
            stride = 1 if i == 0 else 2
            stages.append(_ResNetStage(ic, oc, d, stride, layer_type))
        self.stages = nn.ModuleList(stages)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for stage in self.stages:
            x = stage(op, x)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ResNetModel(nn.Module):
    """ResNet image classification backbone.

    Replicates HuggingFace ``ResNetModel``.

    Inputs: ``pixel_values`` of shape ``(B, C, H, W)``.
    Outputs: Pooled features ``(B, seq_len=1, hidden_size)``.
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: ResNetConfig):
        super().__init__()
        embedding_size = config.embedding_size
        num_channels = getattr(config, "num_channels", 3)

        # Stem: 7x7 conv, stride 2, pad 3 → BN → ReLU → MaxPool 3x3/2
        self.embedder = _ConvBnRelu(
            num_channels,
            embedding_size,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.encoder = _ResNetEncoder(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # Stem: (B, 3, H, W) → (B, embed, H/2, W/2)
        x = self.embedder(op, pixel_values)

        # MaxPool: (B, embed, H/2, W/2) → (B, embed, H/4, W/4)
        x = op.MaxPool(x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

        # Encoder stages: (B, embed, H/4, W/4) → (B, hidden[-1], H/32, W/32)
        x = self.encoder(op, x)

        # Global average pool: (B, C, H', W') → (B, C, 1, 1)
        x = op.GlobalAveragePool(x)

        # Reshape to (B, 1, C) for image-classification task compatibility
        batch = op.Shape(x, start=0, end=1)
        channels = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, [1], channels, axis=0))
        return x

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ResNet weight names to our parameter names.

        HuggingFace nests the stem inside ``embedder.embedder.*``.
        Our model uses a flat ``embedder.*``.
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            new_name = _rename_resnet_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = value
        return new_state_dict


def _rename_resnet_weight(name: str) -> str | None:
    """Rename a single HuggingFace ResNet weight key."""
    # Stem: embedder.embedder.* → embedder.*
    if name.startswith("embedder.embedder."):
        return name[len("embedder.") :]

    # Encoder: encoder.stages.* → encoder.stages.*
    if name.startswith("encoder."):
        return name

    # Drop pooler (we use GlobalAveragePool op instead)
    # Drop classifier head (not in base model)
    return None
