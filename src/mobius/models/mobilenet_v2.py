# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""MobileNet V2 — inverted-residual CNN backbone for mobile image classification.

Architecture overview::

    pixel_values (B, 3, H, W)
      → Stem (MobileNetV2Stem):
          first_conv:  Conv 3x3/2 → BN → ReLU6       (3  → 32 channels)
          conv_3x3:    Depthwise 3x3/1 → BN → ReLU6  (32 depthwise)
          reduce_1x1:  Conv 1x1/1 → BN (no act)       (32 → 16 channels)
      → 16 InvertedResidual layers (expand → depthwise → project)
      → conv_1x1: Conv 1x1 → BN → ReLU6              (320 → 1280 channels)
      → GlobalAveragePool → (B, 1, 1280)

Channel structure (depth_multiplier=1.0)::

    Stem out: 16
    Layer 0-1:   16 → 24   (expand 6x, stride 2/1)
    Layer 2-4:   24 → 32   (expand 6x, stride 2/1/1)
    Layer 5-8:   32 → 64   (expand 6x, stride 2/1/1/1)
    Layer 9-11:  64 → 96   (expand 6x, stride 1/1/1)
    Layer 12-14: 96 → 160  (expand 6x, stride 2/1/1)
    Layer 15:   160 → 320  (expand 6x, stride 1)
    Final:      320 → 1280 (conv_1x1)

TF-style padding is used for 3x3 convolutions: asymmetric [0,1] for stride=2
on even inputs, symmetric [1,1] for stride=1.

Replicates HuggingFace ``MobileNetV2Model``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import MobileNetV2Config
from mobius.components._conv import BatchNorm2d, Conv2dNoBias

if TYPE_CHECKING:
    import onnx_ir as ir


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round v to the nearest multiple of divisor (minimum: min_value)."""
    min_value = divisor if min_value is None else min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _apply_depth_multiplier(v: int, depth_multiplier: float, divisor: int = 8) -> int:
    """Scale channel count by depth_multiplier, rounding to divisor."""
    return _make_divisible(round(v * depth_multiplier), divisor)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnAct(nn.Module):
    """Conv2d (no bias) → BatchNorm2d → optional ReLU6.

    Handles TF-style padding for 3x3 convolutions:
    - stride=2: asymmetric [0, 0, 1, 1] pad (correct for even-sized inputs)
    - stride=1: symmetric [1, 1, 1, 1] pad (SAME)
    - 1x1: no padding needed
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
        layer_norm_eps: float = 0.001,
    ):
        super().__init__()
        # For TF SAME padding, pre-pad before conv with padding=0 in Conv2d.
        # For 3x3/stride=1: symmetric → put padding=1 directly in Conv2d.
        # For 3x3/stride=2: asymmetric → Pad with [0,0,0,0, 0,0,1,1] then Conv2d(padding=0).
        if kernel_size == 3 and stride == 2:
            # Asymmetric TF padding: pad bottom by 1, pad right by 1 (even inputs)
            # ONNX Pad format (4D NCHW): [N_beg, C_beg, H_beg, W_beg, N_end, C_end, H_end, W_end]
            self._pad_values = [0, 0, 0, 0, 0, 0, 1, 1]
            conv_padding = 0
        elif kernel_size == 3 and stride == 1:
            self._pad_values = None
            conv_padding = 1  # symmetric SAME padding via Conv op
        else:
            self._pad_values = None
            conv_padding = 0

        self.convolution = Conv2dNoBias(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            groups=groups,
        )
        self.normalization = BatchNorm2d(out_channels, eps=layer_norm_eps)
        self._activate = activate

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # Apply asymmetric TF padding before stride=2 3x3 conv
        if self._pad_values is not None:
            x = op.Pad(x, op.Constant(value_ints=self._pad_values))
        x = self.convolution(op, x)
        x = self.normalization(op, x)
        if self._activate:
            # ReLU6 = Clip(x, 0, 6)
            x = op.Clip(x, op.Constant(value_float=0.0), op.Constant(value_float=6.0))
        return x


class _MobileNetV2Stem(nn.Module):
    """MobileNetV2 stem block.

    Structure::
        first_conv:  Conv 3x3/2 → BN → ReLU6   (3 → 32 ch)
        conv_3x3:    Depthwise 3x3/1 → BN → ReLU6  (32 depthwise)
        reduce_1x1:  Conv 1x1 → BN (no activation)  (32 → out_ch)

    When ``first_layer_is_expansion=True`` (default), the 1x1 expansion
    step is skipped and the first conv output feeds directly into the depthwise.
    """

    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
    ):
        super().__init__()
        eps = config.layer_norm_eps
        self.first_conv = _ConvBnAct(
            in_channels, expanded_channels, kernel_size=3, stride=2, layer_norm_eps=eps
        )
        self.conv_3x3 = _ConvBnAct(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
            layer_norm_eps=eps,
        )
        self.reduce_1x1 = _ConvBnAct(
            expanded_channels, out_channels, kernel_size=1, activate=False, layer_norm_eps=eps
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.first_conv(op, x)
        x = self.conv_3x3(op, x)
        x = self.reduce_1x1(op, x)
        return x


class _InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    Structure::
        expand_1x1:  Conv 1x1 → BN → ReLU6   (in_ch → expanded_ch)
        conv_3x3:    Depthwise 3x3 → BN → ReLU6   (expanded_ch depthwise)
        reduce_1x1:  Conv 1x1 → BN (linear, no activation)   (expanded_ch → out_ch)

    A residual shortcut is used when stride=1 and in_channels == out_channels
    (linear bottleneck design preserving the information manifold).
    """

    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = 1,
    ):
        super().__init__()
        eps = config.layer_norm_eps
        expanded_channels = _make_divisible(
            round(in_channels * config.expand_ratio), divisor=8
        )
        self._use_residual = stride == 1 and in_channels == out_channels

        self.expand_1x1 = _ConvBnAct(
            in_channels, expanded_channels, kernel_size=1, layer_norm_eps=eps
        )
        # Depthwise conv: groups = expanded_channels
        self.conv_3x3 = _ConvBnAct(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            layer_norm_eps=eps,
        )
        # Linear projection (no activation after)
        self.reduce_1x1 = _ConvBnAct(
            expanded_channels, out_channels, kernel_size=1, activate=False, layer_norm_eps=eps
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        residual = x
        x = self.expand_1x1(op, x)
        x = self.conv_3x3(op, x)
        x = self.reduce_1x1(op, x)
        if self._use_residual:
            x = op.Add(residual, x)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class MobileNetV2Model(nn.Module):
    """MobileNet V2 image classification backbone.

    Replicates HuggingFace ``MobileNetV2Model``.

    Inputs: ``pixel_values`` of shape ``(B, 3, H, W)`` (typically 224x224).
    Outputs: Pooled features ``(B, 1, 1280)`` — compatible with image-classification task.
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: MobileNetV2Config):
        super().__init__()
        dm = config.depth_multiplier

        # Channel sizes after each inverted residual block
        # (length 17: channels[0] = stem output, channels[1..16] = layer outputs)
        raw_channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
        channels = [_apply_depth_multiplier(c, dm) for c in raw_channels]

        # Strides for each of the 16 inverted residual layers
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        # Stem: 3x3/2 + depthwise 3x3/1 + 1x1 projection
        self.conv_stem = _MobileNetV2Stem(
            config,
            in_channels=config.num_channels,
            expanded_channels=_apply_depth_multiplier(32, dm),
            out_channels=channels[0],
        )

        # 16 inverted residual layers — dilated conv not needed for output_stride=32
        current_stride = 2
        dilation = 1
        layers = []
        for i in range(16):
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            layers.append(
                _InvertedResidual(
                    config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                )
            )
        self.layer = nn.ModuleList(layers)

        # Final 1x1 conv expands to 1280 features
        output_channels = _apply_depth_multiplier(1280, dm)
        self.conv_1x1 = _ConvBnAct(
            channels[-1], output_channels, kernel_size=1, layer_norm_eps=config.layer_norm_eps
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # Stem: (B, 3, H, W) → (B, 16, H/2, W/2)
        x = self.conv_stem(op, pixel_values)

        # 16 inverted residual blocks → (B, 320, H/32, W/32)
        for layer in self.layer:
            x = layer(op, x)

        # Final 1x1 expansion: (B, 320, H/32, W/32) → (B, 1280, H/32, W/32)
        x = self.conv_1x1(op, x)

        # Global average pool: (B, 1280, H/32, W/32) → (B, 1280, 1, 1)
        x = op.GlobalAveragePool(x)

        # Reshape to (B, 1, 1280) for image-classification task compatibility
        batch = op.Shape(x, start=0, end=1)
        channels = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, [1], channels, axis=0))
        return x

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace MobileNetV2 weight names to our parameter names.

        HuggingFace nests the model under ``mobilenet_v2.*`` for the
        classification variant; the base model uses flat names matching ours.
        Drops ``num_batches_tracked`` buffers (not used at inference time).
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            # ForImageClassification wraps base model as mobilenet_v2.*
            if name.startswith("mobilenet_v2."):
                name = name[len("mobilenet_v2.") :]
            # Drop inference-unused BN tracking counter
            if name.endswith(".num_batches_tracked"):
                continue
            # Drop classifier head (not part of backbone)
            if name.startswith("classifier."):
                continue
            new_state_dict[name] = value
        return new_state_dict
