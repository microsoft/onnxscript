# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ConvNeXT: a modernized pure-CNN backbone for image classification.

Architecture: Depth-wise separable convolution blocks with LayerNorm,
GELU, inverted bottleneck, and per-channel layer scale.

Reference: https://huggingface.co/docs/transformers/model_doc/convnext
HuggingFace class: ConvNextForImageClassification

Pipeline:
    1. Stem: Conv2d patch embedding (4x4 stride) + LayerNorm
    2. Encoder: 4 stages, each with:
       - Downsampling (stages 1-3): LayerNorm + Conv2d(2x2, stride=2)
       - N ConvNextLayer blocks:
         * Depth-wise Conv2d(7x7, groups=C)
         * Transpose NCHW → NHWC
         * LayerNorm + Linear(C→4C) + GELU + Linear(4C→C)
         * Layer scale (per-channel multiply)
         * Transpose NHWC → NCHW + residual
    3. Global average pool → LayerNorm

Inputs:
    pixel_values: [batch, channels, height, width]

Output:
    last_hidden_state: [batch, 1, hidden_size]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ConvNextConfig
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# ConvNeXT building blocks
# ---------------------------------------------------------------------------


class _ConvNextEmbeddings(nn.Module):
    """Stem: 4x4 patch convolution + LayerNorm (channels-first)."""

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_embeddings = Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.layernorm = LayerNorm(hidden_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # [batch, C_in, H, W] → [batch, C_out, H/4, W/4]
        embeddings = self.patch_embeddings(op, pixel_values)
        # LayerNorm expects channels-last: NCHW → NHWC → norm → NCHW
        embeddings = op.Transpose(embeddings, perm=[0, 2, 3, 1])
        embeddings = self.layernorm(op, embeddings)
        embeddings = op.Transpose(embeddings, perm=[0, 3, 1, 2])
        return embeddings


class _ConvNextLayer(nn.Module):
    """Single ConvNeXT block: depthwise conv → LN → expand → GELU → project → scale → residual.

    Operations alternate between NCHW (for conv) and NHWC (for LN/Linear).
    """

    def __init__(self, hidden_size: int, use_layer_scale: bool):
        super().__init__()
        # Depth-wise 7x7 convolution (groups = channels)
        self.dwconv = Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=hidden_size,
        )
        self.layernorm = LayerNorm(hidden_size)
        # Inverted bottleneck: expand 4x then project back
        self.pwconv1 = Linear(hidden_size, 4 * hidden_size, bias=True)
        self.pwconv2 = Linear(4 * hidden_size, hidden_size, bias=True)
        # Per-channel scale applied before residual
        if use_layer_scale:
            self.layer_scale_parameter = nn.Parameter((hidden_size,))

        self._use_layer_scale = use_layer_scale

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        residual = hidden_states  # NCHW

        # Depth-wise conv in NCHW
        hidden_states = self.dwconv(op, hidden_states)

        # Switch to NHWC for LayerNorm + Linear
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        hidden_states = self.layernorm(op, hidden_states)
        hidden_states = self.pwconv1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.pwconv2(op, hidden_states)

        # Layer scale: element-wise multiply by learned per-channel vector
        if self._use_layer_scale:
            hidden_states = op.Mul(hidden_states, self.layer_scale_parameter)

        # Switch back to NCHW and add residual
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _ConvNextDownsample(nn.Module):
    """Spatial downsampling between stages: LayerNorm + Conv2d(2x2, stride=2)."""

    def __init__(self, in_channels: int, out_channels: int, eps: float):
        super().__init__()
        self.layernorm = LayerNorm(in_channels, eps=eps)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # LayerNorm in channels-last, then conv in NCHW
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        hidden_states = self.layernorm(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = self.conv(op, hidden_states)
        return hidden_states


class _ConvNextStage(nn.Module):
    """One encoder stage: optional downsampling + N ConvNextLayer blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        use_layer_scale: bool,
        eps: float,
        downsample: bool,
    ):
        super().__init__()
        if downsample:
            self.downsampling_layer = _ConvNextDownsample(in_channels, out_channels, eps)
        self.layers = nn.ModuleList(
            [_ConvNextLayer(out_channels, use_layer_scale) for _ in range(depth)]
        )
        self._downsample = downsample

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        if self._downsample:
            hidden_states = self.downsampling_layer(op, hidden_states)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ConvNextModel(nn.Module):
    """ConvNeXT image classification backbone.

    A pure-CNN architecture that modernizes ResNet with:
    - Depth-wise separable convolutions
    - LayerNorm instead of BatchNorm
    - GELU activation
    - Inverted bottleneck (expand 4x, project back)
    - Per-channel layer scale
    - 4x4 patch stem (like ViT)

    HuggingFace: ``ConvNextForImageClassification``
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: ConvNextConfig):
        super().__init__()
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        eps = config.rms_norm_eps
        use_layer_scale = config.layer_scale_init_value > 0

        # Stem: patch embedding
        self.embeddings = _ConvNextEmbeddings(
            config.num_channels, hidden_sizes[0], config.patch_size
        )

        # Encoder: 4 stages
        stages = []
        for i in range(len(depths)):
            in_ch = hidden_sizes[i - 1] if i > 0 else hidden_sizes[0]
            out_ch = hidden_sizes[i]
            stages.append(
                _ConvNextStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    depth=depths[i],
                    use_layer_scale=use_layer_scale,
                    eps=eps,
                    downsample=(i > 0),
                )
            )
        self.encoder = nn.ModuleList(stages)

        # Final layer norm
        self.layernorm = LayerNorm(hidden_sizes[-1], eps=eps)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        """Forward pass: image → pooled feature vector.

        Args:
            pixel_values: (batch, channels, H, W)

        Returns:
            (batch, 1, hidden_size) — global average pooled features.
        """
        hidden_states = self.embeddings(op, pixel_values)

        for stage in self.encoder:
            hidden_states = stage(op, hidden_states)

        # Global average pooling: NCHW → mean over H,W → (batch, C)
        pooled = op.ReduceMean(hidden_states, axes=[2, 3], keepdims=False)

        # LayerNorm on the pooled vector
        pooled = self.layernorm(op, pooled)

        # Reshape to (batch, 1, hidden_size) for image-classification task
        batch_size = op.Shape(pooled, start=0, end=1)
        hidden_dim = op.Shape(pooled, start=1, end=2)
        pooled = op.Reshape(pooled, op.Concat(batch_size, [1], hidden_dim, axis=0))
        return pooled

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_convnext_weight(name)
            if new_name is not None:
                new_dict[new_name] = tensor
        return new_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------


def _rename_convnext_weight(name: str) -> str | None:
    """Map HuggingFace ConvNeXT weight names to our naming convention."""
    # Strip model prefix
    for prefix in ("convnext.", "model.convnext."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Skip classifier weights (not part of the backbone)
    if name.startswith("classifier."):
        return None

    # Embeddings: pass through
    if name.startswith("embeddings."):
        return name

    # Final layernorm
    if name.startswith("layernorm."):
        return name

    # Encoder stages
    if name.startswith("encoder.stages."):
        parts = name.split(".", 3)  # encoder, stages, idx, remainder
        if len(parts) < 4:
            return None
        stage_idx = parts[2]
        remainder = parts[3]

        # Downsampling layer: stages.{s}.downsampling_layer.{0,1}.X
        # HF: downsampling_layer is a ModuleList [LayerNorm, Conv2d]
        # Ours: downsampling_layer is _ConvNextDownsample with .layernorm + .conv
        if remainder.startswith("downsampling_layer."):
            dl_rest = remainder[len("downsampling_layer.") :]
            if dl_rest.startswith("0."):
                # Index 0 = LayerNorm → .layernorm
                return f"encoder.{stage_idx}.downsampling_layer.layernorm.{dl_rest[2:]}"
            if dl_rest.startswith("1."):
                # Index 1 = Conv2d → .conv
                return f"encoder.{stage_idx}.downsampling_layer.conv.{dl_rest[2:]}"
            return None

        # Layer weights: stages.{s}.layers.{l}.X → encoder.{s}.layers.{l}.X
        if remainder.startswith("layers."):
            return f"encoder.{stage_idx}.{remainder}"

        return None

    return None
