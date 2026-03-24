# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything model for monocular depth estimation.

Depth Anything uses a DINOv2 (ViT) backbone with a DPT (Dense Prediction
Transformer) decoder head. The backbone extracts multi-scale features at
specific layer indices, which are then reassembled, fused, and decoded
into a per-pixel depth map.
"""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, DepthAnythingConfig
from mobius.components import (
    FCMLP,
    Conv2d,
    Conv2dNoBias,
    ConvTranspose2d,
    EncoderAttention,
    LayerNorm,
)

# ---------------------------------------------------------------------------
# Backbone: ViT encoder that returns hidden states at specific layers
# ---------------------------------------------------------------------------


class _ViTBackbone(nn.Module):
    """ViT backbone that extracts hidden states at specified layer indices."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, hidden_size, patch_size)
        self.cls_token = nn.Parameter(
            [1, 1, hidden_size],
            data=ir.tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)),
        )
        self.position_embeddings = nn.Parameter(
            [1, num_patches + 1, hidden_size],
            data=ir.tensor(np.zeros((1, num_patches + 1, hidden_size), dtype=np.float32)),
        )
        self.encoder = nn.ModuleList(
            [_ViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 1-indexed out_indices (layer 1 = after first layer)
        self.out_indices = config.backbone_out_indices or []

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, op.Constant(value_ints=[1, 1]), axis=0),
        )
        hidden_states = op.Concat(cls_tokens, patch_embeds, axis=1)
        hidden_states = op.Add(hidden_states, self.position_embeddings)

        feature_maps = []
        for i, layer in enumerate(self.encoder):
            hidden_states = layer(op, hidden_states)
            # out_indices are 1-indexed: index 1 = after layer 0
            if (i + 1) in self.out_indices:
                normed = self.layernorm(op, hidden_states)
                feature_maps.append(normed)

        return feature_maps


class _Conv2dPatchEmbed(nn.Module):
    """Conv2d-based patch embedding."""

    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = Conv2d(in_channels, hidden_size, patch_size, patch_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        x = self.projection(op, pixel_values)
        batch = op.Shape(x, start=0, end=1)
        hidden = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, hidden, op.Constant(value_ints=[-1]), axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class _ViTEncoderLayer(nn.Module):
    """ViT pre-norm encoder layer."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layernorm_before = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = EncoderAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            bias=True,
        )
        self.layernorm_after = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.layernorm_before(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layernorm_after(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# DPT Neck: Reassemble + Fusion
# ---------------------------------------------------------------------------


class _ReassembleLayer(nn.Module):
    """Reassemble: project channels and resize spatially."""

    def __init__(self, in_channels: int, out_channels: int, factor: float):
        super().__init__()
        self.projection = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self._factor = factor
        if factor > 1:
            self.resize = ConvTranspose2d(
                out_channels, out_channels, kernel_size=int(factor), stride=int(factor)
            )
        elif factor < 1:
            self.resize = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=int(1 / factor), padding=1
            )
        else:
            self.resize = None  # Identity

    def forward(self, op: builder.OpBuilder, hidden_state: ir.Value):
        hidden_state = self.projection(op, hidden_state)
        if self.resize is not None:
            hidden_state = self.resize(op, hidden_state)
        return hidden_state


class _PreActResidualLayer(nn.Module):
    """Pre-activation residual conv block: ReLU → Conv → ReLU → Conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.convolution1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.convolution2 = Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, op: builder.OpBuilder, hidden_state: ir.Value):
        residual = hidden_state
        hidden_state = op.Relu(hidden_state)
        hidden_state = self.convolution1(op, hidden_state)
        hidden_state = op.Relu(hidden_state)
        hidden_state = self.convolution2(op, hidden_state)
        return op.Add(hidden_state, residual)


class _FeatureFusionLayer(nn.Module):
    """Fuse two feature maps with residual blocks and upsample."""

    def __init__(self, channels: int):
        super().__init__()
        self.projection = Conv2d(channels, channels, kernel_size=1, padding=0)
        self.residual_layer1 = _PreActResidualLayer(channels)
        self.residual_layer2 = _PreActResidualLayer(channels)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_state: ir.Value,
        residual: ir.Value | None = None,
        scale_factor: int = 2,
    ):
        if residual is not None:
            # Resize residual to match hidden_state if needed
            target_h = op.Shape(hidden_state, start=2, end=3)
            target_w = op.Shape(hidden_state, start=3, end=4)
            residual = op.Resize(
                residual,
                None,  # roi (empty)
                None,  # scales (empty)
                op.Concat(
                    op.Shape(residual, start=0, end=2),
                    target_h,
                    target_w,
                    axis=0,
                ),
                mode="linear",
            )
            residual = self.residual_layer1(op, residual)
            hidden_state = op.Add(hidden_state, residual)

        hidden_state = self.residual_layer2(op, hidden_state)

        # Upsample by scale_factor
        hidden_state = op.Resize(
            hidden_state,
            None,  # roi
            op.Constant(value_floats=[1.0, 1.0, float(scale_factor), float(scale_factor)]),
            mode="linear",
        )
        hidden_state = self.projection(op, hidden_state)
        return hidden_state


class _DepthAnythingNeck(nn.Module):
    """DPT neck: reassemble multi-scale features and fuse them."""

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()
        neck_sizes = config.neck_hidden_sizes or [48, 96, 192, 384]
        factors = config.reassemble_factors or [4.0, 2.0, 1.0, 0.5]
        fusion_size = config.fusion_hidden_size
        backbone_hidden = config.hidden_size

        self.reassemble_layers = nn.ModuleList(
            [_ReassembleLayer(backbone_hidden, ch, f) for ch, f in zip(neck_sizes, factors)]
        )
        self.convs = nn.ModuleList(
            [Conv2dNoBias(ch, fusion_size, kernel_size=3, padding=1) for ch in neck_sizes]
        )
        self.fusion_layers = nn.ModuleList(
            [_FeatureFusionLayer(fusion_size) for _ in neck_sizes]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        patch_height: ir.Value,
        patch_width: ir.Value,
    ):
        # Reassemble: reshape from [B, S+1, H] to [B, C, pH, pW], project, resize
        reassembled = []
        for i, hs in enumerate(hidden_states):
            # Remove CLS token: [B, S+1, H] -> [B, S, H]
            hs = op.Slice(
                hs,
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2**31 - 1]),
                op.Constant(value_ints=[1]),  # axis=1
            )
            # Reshape to [B, H, pH, pW]
            batch = op.Shape(hs, start=0, end=1)
            channels = op.Shape(hs, start=2, end=3)
            hs = op.Transpose(hs, perm=[0, 2, 1])
            hs = op.Reshape(
                hs,
                op.Concat(batch, channels, patch_height, patch_width, axis=0),
            )
            hs = self.reassemble_layers[i](op, hs)
            hs = self.convs[i](op, hs)
            reassembled.append(hs)

        # Fusion: coarse-to-fine (reverse order)
        reassembled.reverse()
        fused = None
        fused_list = []
        for feature, layer in zip(reassembled, self.fusion_layers):
            if fused is None:
                fused = layer(op, feature)
            else:
                fused = layer(op, fused, feature)
            fused_list.append(fused)

        return fused_list


class _DepthEstimationHead(nn.Module):
    """Depth prediction head: 3 Conv2d layers with upsample."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        fusion_size = config.fusion_hidden_size
        head_size = config.head_hidden_size
        self._patch_size = config.patch_size

        self.conv1 = Conv2d(fusion_size, fusion_size // 2, kernel_size=3, padding=1)
        self.conv2 = Conv2d(fusion_size // 2, head_size, kernel_size=3, padding=1)
        self.conv3 = Conv2d(head_size, 1, kernel_size=1, padding=0)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        patch_height: ir.Value,
        patch_width: ir.Value,
    ):
        # Use the last fused feature map
        x = hidden_states[-1]

        x = self.conv1(op, x)
        # Upsample to original resolution
        target_h = op.Mul(patch_height, op.Constant(value_int=self._patch_size))
        target_w = op.Mul(patch_width, op.Constant(value_int=self._patch_size))
        x = op.Resize(
            x,
            None,  # roi
            None,  # scales
            op.Concat(
                op.Shape(x, start=0, end=2),
                target_h,
                target_w,
                axis=0,
            ),
            mode="linear",
        )
        x = self.conv2(op, x)
        x = op.Relu(x)
        x = self.conv3(op, x)
        x = op.Relu(x)
        # Squeeze channel dim: [B, 1, H, W] → [B, H, W]
        x = op.Squeeze(x, [1])
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DepthAnythingForDepthEstimation(nn.Module):
    """Depth Anything model for monocular depth estimation.

    Uses a DINOv2 (ViT) backbone with a DPT decoder for dense depth prediction.
    """

    default_task = "image-classification"
    category = "Depth Estimation"
    config_class: type = DepthAnythingConfig

    def __init__(self, config: DepthAnythingConfig):
        super().__init__()
        self.config = config
        self._patch_size = config.patch_size

        self.backbone = _ViTBackbone(config)
        self.neck = _DepthAnythingNeck(config)
        self.head = _DepthEstimationHead(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        feature_maps = self.backbone(op, pixel_values)

        patch_height = op.Constant(value_int=self.config.image_size // self._patch_size)
        patch_width = op.Constant(value_int=self.config.image_size // self._patch_size)
        # Reshape for Concat compatibility
        patch_height = op.Reshape(patch_height, op.Constant(value_ints=[1]))
        patch_width = op.Reshape(patch_width, op.Constant(value_ints=[1]))

        fused = self.neck(op, feature_maps, patch_height, patch_width)
        depth = self.head(op, fused, patch_height, patch_width)
        return depth

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_depth_anything_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_BACKBONE_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
}

_BACKBONE_LAYER_PATTERN = re.compile(r"^backbone\.encoder\.layer\.(\d+)\.(.+)$")


def _rename_depth_anything_weight(name: str) -> str | None:
    """Rename HF Depth Anything weight to our naming convention."""
    # Backbone embeddings
    if name == "backbone.embeddings.cls_token":
        return "backbone.cls_token"
    if name == "backbone.embeddings.position_embeddings":
        return "backbone.position_embeddings"
    if name.startswith("backbone.embeddings.patch_embeddings.projection."):
        suffix = name[len("backbone.embeddings.patch_embeddings.projection.") :]
        return f"backbone.patch_embeddings.projection.{suffix}"

    # Backbone layernorm
    if name.startswith("backbone.layernorm."):
        suffix = name[len("backbone.layernorm.") :]
        return f"backbone.layernorm.{suffix}"

    # Backbone encoder layers
    m = _BACKBONE_LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        if suffix.startswith("layernorm_"):
            return f"backbone.encoder.{layer_idx}.{suffix}"
        for old, new in _BACKBONE_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"backbone.encoder.{layer_idx}.{new}{remainder}"
        return None

    # Neck reassemble
    if name.startswith("neck.reassemble_stage.layers."):
        suffix = name[len("neck.reassemble_stage.layers.") :]
        return f"neck.reassemble_layers.{suffix}"

    # Neck convs
    if name.startswith("neck.convs."):
        return name

    # Neck fusion
    if name.startswith("neck.fusion_stage.layers."):
        suffix = name[len("neck.fusion_stage.layers.") :]
        return f"neck.fusion_layers.{suffix}"

    # Head
    if name.startswith("head."):
        return name

    # Skip unknown
    return None
