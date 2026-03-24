# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ViT (Vision Transformer) model for image feature extraction."""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    EncoderAttention,
    LayerNorm,
)
from mobius.components import (
    Conv2d as _Conv2d,
)


class ViTModel(nn.Module):
    """Vision Transformer for image feature extraction.

    Pre-norm encoder with patch embeddings, CLS token, and learned
    position embeddings. Output is the last hidden state including the
    CLS token at position 0.
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        image_size = getattr(config, "image_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        num_channels = getattr(config, "num_channels", 3)
        num_patches = (image_size // patch_size) ** 2

        self.embeddings = _ViTEmbeddings(config, num_patches, patch_size, num_channels)
        self.encoder = _ViTEncoder(config)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.encoder(op, hidden_states)
        hidden_states = self.layernorm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_vit_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


class _ViTEmbeddings(nn.Module):
    """ViT embeddings: Conv2d patch embed + CLS token + position embeddings."""

    def __init__(self, config, num_patches, patch_size, num_channels):
        super().__init__()
        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, config.hidden_size, patch_size)
        # CLS token and position embeddings as parameters with pre-computed data
        self.cls_token = nn.Parameter(
            [1, 1, config.hidden_size],
            data=ir.tensor(np.zeros((1, 1, config.hidden_size), dtype=np.float32)),
        )
        self.position_embeddings = nn.Parameter(
            [1, num_patches + 1, config.hidden_size],
            data=ir.tensor(
                np.zeros((1, num_patches + 1, config.hidden_size), dtype=np.float32)
            ),
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        # Expand CLS token to batch size
        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, op.Constant(value_ints=[1, 1]), axis=0),
        )
        # Prepend CLS token to patch embeddings
        hidden_states = op.Concat(cls_tokens, patch_embeds, axis=1)
        # Add position embeddings
        hidden_states = op.Add(hidden_states, self.position_embeddings)
        return hidden_states


class _Conv2dPatchEmbed(nn.Module):
    """Conv2d-based patch embedding."""

    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = _Conv2d(in_channels, hidden_size, patch_size, patch_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Conv2d: [batch, channels, H, W] -> [batch, hidden, H/patch, W/patch]
        x = self.projection(op, pixel_values)
        # Flatten spatial dims and transpose: [batch, hidden, num_patches] -> [batch, num_patches, hidden]
        batch = op.Shape(x, start=0, end=1)
        hidden = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, hidden, op.Constant(value_ints=[-1]), axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class _ViTEncoder(nn.Module):
    """ViT encoder: pre-norm encoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [_ViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for layer in self.layer:
            hidden_states = layer(op, hidden_states)
        return hidden_states


class _ViTEncoderLayer(nn.Module):
    """ViT pre-norm encoder layer: norm → attn → residual → norm → mlp → residual."""

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


# Weight name mapping
_VIT_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
}

_LAYER_PATTERN = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")


def _rename_vit_weight(name: str) -> str | None:
    """Rename HF ViT weight to our naming convention."""
    # Strip vit. prefix if present
    if name.startswith("vit."):
        name = name[4:]

    # Skip pooler and classifier heads
    if name.startswith(("pooler.", "classifier.")):
        return None

    # Embeddings
    if name == "embeddings.cls_token":
        return "embeddings.cls_token"
    if name == "embeddings.position_embeddings":
        return "embeddings.position_embeddings"
    if name.startswith("embeddings.patch_embeddings.projection."):
        suffix = name[len("embeddings.patch_embeddings.projection.") :]
        return f"embeddings.patch_embeddings.projection.{suffix}"

    # Final layernorm
    if name.startswith("layernorm."):
        return name  # Already correct

    # Encoder layers
    m = _LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        # layernorm_before / layernorm_after pass through
        if suffix.startswith("layernorm_"):
            return f"encoder.layer.{layer_idx}.{suffix}"
        for old, new in _VIT_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"encoder.layer.{layer_idx}.{new}{remainder}"

    return None
