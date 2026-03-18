# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""YOLOS (You Only Look at One Sequence) model for object detection.

YOLOS extends ViT for detection by appending learnable detection tokens
to the patch sequence and adding MLP prediction heads for class labels
and bounding boxes. Mid-position embeddings are added between encoder
layers for improved detection.
"""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, YolosConfig
from mobius.components import (
    Conv2d as _Conv2d,
)
from mobius.components import (
    EncoderAttention,
    LayerNorm,
    Linear,
)


class _YolosEmbeddings(nn.Module):
    """YOLOS embeddings: patch embed + CLS token + detection tokens + position."""

    def __init__(self, config: YolosConfig):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_detection_tokens = config.num_detection_tokens

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches + num_detection_tokens + 1  # patches + det + CLS

        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, hidden_size, patch_size)
        self.cls_token = nn.Parameter(
            [1, 1, hidden_size],
            data=ir.tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)),
        )
        self.detection_tokens = nn.Parameter(
            [1, num_detection_tokens, hidden_size],
            data=ir.tensor(np.zeros((1, num_detection_tokens, hidden_size), dtype=np.float32)),
        )
        self.position_embeddings = nn.Parameter(
            [1, num_positions, hidden_size],
            data=ir.tensor(np.zeros((1, num_positions, hidden_size), dtype=np.float32)),
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        # Expand CLS and detection tokens to batch size
        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, op.Constant(value_ints=[1, 1]), axis=0),
        )
        det_tokens = op.Expand(
            self.detection_tokens,
            op.Concat(
                batch_size,
                op.Shape(self.detection_tokens, start=1, end=3),
                axis=0,
            ),
        )
        # [CLS, patches, detection_tokens]
        hidden_states = op.Concat(cls_tokens, patch_embeds, det_tokens, axis=1)
        # Add position embeddings (skip interpolation for fixed-size export)
        hidden_states = op.Add(hidden_states, self.position_embeddings)
        return hidden_states


class _Conv2dPatchEmbed(nn.Module):
    """Conv2d-based patch embedding."""

    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = _Conv2d(in_channels, hidden_size, patch_size, patch_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        x = self.projection(op, pixel_values)
        batch = op.Shape(x, start=0, end=1)
        hidden = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, hidden, op.Constant(value_ints=[-1]), axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class _YolosEncoderLayer(nn.Module):
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
        self.mlp = _YolosMLP(config)

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


class _YolosMLP(nn.Module):
    """ViT MLP: Linear → GELU → Linear."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        from mobius.components._activations import ACT2FN

        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.up_proj(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        return self.down_proj(op, hidden_states)


class _MLPPredictionHead(nn.Module):
    """DETR-style MLP prediction head (3 layers with ReLU)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [Linear(dims[i], dims[i + 1], bias=True) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        for i, layer in enumerate(self.layers):
            x = layer(op, x)
            if i < self.num_layers - 1:
                x = op.Relu(x)
        return x


class YolosForObjectDetection(nn.Module):
    """YOLOS model for object detection.

    Outputs class logits and bounding box predictions from detection tokens.
    """

    default_task = "object-detection"
    category = "Object Detection"
    config_class: type = YolosConfig

    def __init__(self, config: YolosConfig):
        super().__init__()
        self.config = config
        self.num_detection_tokens = config.num_detection_tokens

        # ViT backbone
        self.embeddings = _YolosEmbeddings(config)
        self.encoder = nn.ModuleList(
            [_YolosEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Mid-position embeddings (added between encoder layers)
        num_patches = (config.image_size // config.patch_size) ** 2
        seq_length = 1 + num_patches + config.num_detection_tokens
        if config.num_hidden_layers > 1:
            self.mid_position_embeddings = nn.Parameter(
                [config.num_hidden_layers - 1, 1, seq_length, config.hidden_size],
                data=ir.tensor(
                    np.zeros(
                        (config.num_hidden_layers - 1, 1, seq_length, config.hidden_size),
                        dtype=np.float32,
                    )
                ),
            )
        else:
            self.mid_position_embeddings = None

        # Detection heads
        self.class_labels_classifier = _MLPPredictionHead(
            config.hidden_size, config.hidden_size, config.num_labels + 1, num_layers=3
        )
        self.bbox_predictor = _MLPPredictionHead(
            config.hidden_size, config.hidden_size, 4, num_layers=3
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)

        for i, layer in enumerate(self.encoder):
            hidden_states = layer(op, hidden_states)
            # Add mid-position embeddings between layers
            if self.mid_position_embeddings is not None and i < len(self.encoder) - 1:
                mid_embed = op.Gather(
                    self.mid_position_embeddings, op.Constant(value_int=i), axis=0
                )
                hidden_states = op.Add(hidden_states, mid_embed)

        hidden_states = self.layernorm(op, hidden_states)

        # Extract detection token outputs (last num_detection_tokens positions)
        det_output = op.Slice(
            hidden_states,
            op.Constant(value_ints=[-self.num_detection_tokens]),
            op.Constant(value_ints=[2**31 - 1]),  # INT_MAX as end
            op.Constant(value_ints=[1]),  # axis
        )

        logits = self.class_labels_classifier(op, det_output)
        pred_boxes = op.Sigmoid(self.bbox_predictor(op, det_output))

        return logits, pred_boxes

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_yolos_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


_YOLOS_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
}

_LAYER_PATTERN = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")


def _rename_yolos_weight(name: str) -> str | None:
    """Rename HF YOLOS weight to our naming convention."""
    # Strip vit. prefix
    if name.startswith("vit."):
        name = name[4:]

    # Skip pooler
    if name.startswith("pooler."):
        return None

    # Embeddings
    if name == "embeddings.cls_token":
        return "embeddings.cls_token"
    if name == "embeddings.detection_tokens":
        return "embeddings.detection_tokens"
    if name == "embeddings.position_embeddings":
        return "embeddings.position_embeddings"
    if name.startswith("embeddings.patch_embeddings.projection."):
        suffix = name[len("embeddings.patch_embeddings.projection.") :]
        return f"embeddings.patch_embeddings.projection.{suffix}"

    # Final layernorm
    if name.startswith("layernorm."):
        return name

    # Mid-position embeddings
    if name == "encoder.mid_position_embeddings":
        return "mid_position_embeddings"

    # Detection heads (pass through)
    if name.startswith(("class_labels_classifier.", "bbox_predictor.")):
        return name

    # Encoder layers
    m = _LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        if suffix.startswith("layernorm_"):
            return f"encoder.{layer_idx}.{suffix}"
        for old, new in _YOLOS_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"encoder.{layer_idx}.{new}{remainder}"

    return None
