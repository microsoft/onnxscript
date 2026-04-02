# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DETR (DEtection TRansformer) and table-transformer object detection models.

DETR combines a ResNet-50 CNN backbone with a transformer encoder-decoder
to perform end-to-end object detection without hand-crafted anchors or NMS.

Architecture::

    pixel_values (B, 3, H, W)
      → ResNet-50 backbone (no GAP)        → (B, 2048, H/32, W/32)
      → 1x1 Conv input projection           → (B, d_model, H/32, W/32)
      → Flatten + sine position encoding    → (B, H'*W', d_model)
      → Transformer encoder (6 layers)      → memory (B, H'*W', d_model)
      → Transformer decoder (6 layers)      → (B, num_queries, d_model)
              ← object query positions (learned, (num_queries, d_model))
      → class head (Linear)                 → (B, num_queries, num_labels+1)
      → bbox head (3-layer MLP + sigmoid)   → (B, num_queries, 4)

Post-norm (BERT-style) transformer: residual add then LayerNorm.
Position embeddings are added to Q and K inputs *before* projection in
encoder self-attention and decoder cross-attention.

table-transformer (``microsoft/table-transformer-detection``) uses the same
architecture as DETR with different fine-tuned weights.

Replicates HuggingFace ``DetrForObjectDetection``.
"""

from __future__ import annotations

import math
import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import DetrConfig, ResNetConfig
from mobius.components import Conv2d, LayerNorm, Linear
from mobius.models.resnet import _ConvBnRelu, _ResNetEncoder

# ---------------------------------------------------------------------------
# Sinusoidal 2D position encoding (fixed, precomputed as a constant)
# ---------------------------------------------------------------------------


def _compute_sine_pos_embed(h: int, w: int, d_model: int = 256) -> np.ndarray:
    """Compute 2D sinusoidal positional encoding for a fixed feature map size.

    Replicates HuggingFace ``DetrSinePositionEmbedding.forward`` for a
    batch of all-unmasked (full-image) inputs.  The result is stored as a
    constant ONNX initializer so no runtime recomputation is needed.

    Args:
        h: Feature map height (image_size // 32 for stride-32 ResNet-50).
        w: Feature map width.
        d_model: Transformer hidden dimension (split equally between y and x).

    Returns:
        Float32 array of shape ``(1, h*w, d_model)``.
    """
    num_pos_feats = d_model // 2
    temperature = 10000.0
    scale = 2.0 * math.pi
    eps = 1e-6

    # Row and column indices 1..h and 1..w (matches HF cumsum over unmasked grid)
    y_embed = np.arange(1, h + 1, dtype=np.float32).reshape(h, 1).repeat(w, axis=1)
    x_embed = np.arange(1, w + 1, dtype=np.float32).reshape(1, w).repeat(h, axis=0)

    # Normalize to [0, 2π]
    y_embed = y_embed / (h + eps) * scale
    x_embed = x_embed / (w + eps) * scale

    # Frequency bands: adjacent pairs share the same frequency (sin/cos pair)
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2.0 * (dim_t // 2) / num_pos_feats)

    # (h, w, num_pos_feats) — angle = position / frequency
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    # Even indices → sin, odd → cos; interleave: (h, w, num_pos_feats)
    pos_x = np.stack([np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])], axis=3).reshape(
        h, w, num_pos_feats
    )
    pos_y = np.stack([np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])], axis=3).reshape(
        h, w, num_pos_feats
    )

    # Concat y and x halves, flatten spatial dims: (1, h*w, d_model)
    pos = np.concatenate([pos_y, pos_x], axis=2)  # (h, w, d_model)
    return pos.reshape(1, h * w, d_model).astype(np.float32)


# ---------------------------------------------------------------------------
# ResNet backbone (no global average pool)
# ---------------------------------------------------------------------------


class _DetrResNetBackbone(nn.Module):
    """ResNet backbone for DETR — returns raw C4 spatial feature map.

    Unlike :class:`~mobius.models.resnet.ResNetModel` which pools to
    ``(B, 1, C)``, this variant returns ``(B, C, H', W')`` where
    H' = H//32, W' = W//32 (stride-32 ResNet-50 with stride-2 pooling in stem).
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        # Build a ResNetConfig from the backbone fields embedded in DetrConfig
        rc = ResNetConfig(
            embedding_size=config.backbone_embedding_size,
            hidden_sizes=config.backbone_hidden_sizes,
            depths=config.backbone_depths,
            layer_type=config.backbone_layer_type,
        )
        num_channels = getattr(config, "num_channels", 3)
        # Stem: 7x7 conv stride-2 + BN + ReLU
        self.embedder = _ConvBnRelu(
            num_channels,
            rc.embedding_size,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.encoder = _ResNetEncoder(rc)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # Stem: (B, 3, H, W) → (B, embed, H/2, W/2)
        x = self.embedder(op, pixel_values)
        # MaxPool: (B, embed, H/2, W/2) → (B, embed, H/4, W/4)
        x = op.MaxPool(x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        # Encoder stages: (B, embed, H/4, W/4) → (B, hidden[-1], H/32, W/32)
        return self.encoder(op, x)  # NO GlobalAveragePool


# ---------------------------------------------------------------------------
# DETR attention (position embeddings added to Q/K before projection)
# ---------------------------------------------------------------------------


class _DetrAttention(nn.Module):
    """Multi-head attention with optional positional bias on Q and K inputs.

    DETR adds spatial position embeddings to the *inputs* of the Q and K
    projections (``with_pos_embed`` in HuggingFace), not to the projected
    outputs.  V always uses the raw hidden states without any position.

    This is used for:
    - Encoder self-attention: both Q and K get the spatial pos embed.
    - Decoder self-attention: both Q and K get the object-query pos embed.
    - Decoder cross-attention: Q gets query pos, K gets spatial pos.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
        q_pos: ir.Value | None = None,
        k_pos: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        # Cross-attention uses encoder hidden states for K/V; self-attn uses hidden_states
        kv_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        # Add positional embeddings to Q/K inputs before projection (broadcast over batch)
        q_input = op.Add(hidden_states, q_pos) if q_pos is not None else hidden_states
        k_input = op.Add(kv_states, k_pos) if k_pos is not None else kv_states

        # Project to query/key/value representations: (B, seq, d_model)
        query = self.q_proj(op, q_input)
        key = self.k_proj(op, k_input)
        value = self.v_proj(op, kv_states)  # V: no position

        # Scaled dot-product attention via ONNX Attention op
        attn_output = op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )
        return self.out_proj(op, attn_output)


# ---------------------------------------------------------------------------
# Feed-forward network (named fc1/fc2 to match HuggingFace weight names)
# ---------------------------------------------------------------------------


class _DetrFFN(nn.Module):
    """Two-layer MLP with ReLU, matching HuggingFace ``DetrEncoderLayer.mlp``."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.fc2(op, op.Relu(self.fc1(op, x)))


# ---------------------------------------------------------------------------
# Encoder layer: self-attn + post-norm + FFN + post-norm
# ---------------------------------------------------------------------------


class _DetrEncoderLayer(nn.Module):
    """DETR encoder layer.

    Structure (post-norm like BERT)::

        hidden_states + self_attn(hidden_states, pos) → LayerNorm
        + mlp(hidden_states)                          → LayerNorm
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        d = config.d_model
        self.self_attn = _DetrAttention(d, config.encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(d, eps=1e-5)
        self.mlp = _DetrFFN(d, config.encoder_ffn_dim)
        self.final_layer_norm = LayerNorm(d, eps=1e-5)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        pos_embed: ir.Value,
    ) -> ir.Value:
        # Self-attention: spatial pos added to Q and K
        residual = hidden_states
        attn_out = self.self_attn(op, hidden_states, q_pos=pos_embed, k_pos=pos_embed)
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, attn_out))

        # FFN
        residual = hidden_states
        mlp_out = self.mlp(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, mlp_out))

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer: self-attn + cross-attn + FFN (all post-norm)
# ---------------------------------------------------------------------------


class _DetrDecoderLayer(nn.Module):
    """DETR decoder layer.

    Structure (post-norm)::

        hidden_states + self_attn(hidden_states, query_pos)     → LayerNorm
        + encoder_attn(hidden_states, memory, query_pos, pos)   → LayerNorm
        + mlp(hidden_states)                                    → LayerNorm
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        d = config.d_model
        self.self_attn = _DetrAttention(d, config.decoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(d, eps=1e-5)
        self.encoder_attn = _DetrAttention(d, config.decoder_attention_heads)
        self.encoder_attn_layer_norm = LayerNorm(d, eps=1e-5)
        self.mlp = _DetrFFN(d, config.decoder_ffn_dim)
        self.final_layer_norm = LayerNorm(d, eps=1e-5)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        query_pos: ir.Value,
        spatial_pos: ir.Value,
    ) -> ir.Value:
        # Self-attention: object queries attend to each other with query positional bias
        residual = hidden_states
        attn_out = self.self_attn(op, hidden_states, q_pos=query_pos, k_pos=query_pos)
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, attn_out))

        # Cross-attention: queries attend to encoder memory
        # Q gets query pos embed, K gets spatial pos embed
        residual = hidden_states
        attn_out = self.encoder_attn(
            op,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            q_pos=query_pos,
            k_pos=spatial_pos,
        )
        hidden_states = self.encoder_attn_layer_norm(op, op.Add(residual, attn_out))

        # FFN
        residual = hidden_states
        mlp_out = self.mlp(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, mlp_out))

        return hidden_states


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------


class _DetrEncoder(nn.Module):
    """Stack of DETR encoder layers."""

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DetrEncoderLayer(config) for _ in range(config.encoder_layers)]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        pos_embed: ir.Value,
    ) -> ir.Value:
        # Pass spatial pos embed into each encoder layer
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, pos_embed)
        return hidden_states


class _DetrDecoder(nn.Module):
    """Stack of DETR decoder layers followed by a final LayerNorm."""

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DetrDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm = LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        query_pos: ir.Value,
        spatial_pos: ir.Value,
    ) -> ir.Value:
        for layer in self.layers:
            hidden_states = layer(
                op, hidden_states, encoder_hidden_states, query_pos, spatial_pos
            )
        return self.layernorm(op, hidden_states)


# ---------------------------------------------------------------------------
# MLP prediction head (3-layer with ReLU between layers)
# ---------------------------------------------------------------------------


class _MLPPredictionHead(nn.Module):
    """DETR-style MLP prediction head for bounding box regression.

    Three Linear layers with ReLU between them and no final activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [Linear(dims[i], dims[i + 1], bias=True) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for i, layer in enumerate(self.layers):
            x = layer(op, x)
            if i < self.num_layers - 1:
                x = op.Relu(x)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class DetrForObjectDetection(nn.Module):
    """DETR (DEtection TRansformer) for object detection.

    Combines a ResNet-50 CNN backbone with a transformer encoder-decoder to
    perform end-to-end object detection without hand-crafted anchors or NMS.
    Outputs class logits and bounding-box predictions for ``num_queries``
    object query slots (default 100).

    Also used for ``table-transformer`` (``microsoft/table-transformer-detection``)
    which shares the same architecture with different fine-tuned weights.

    Inputs:
        ``pixel_values``: ``(B, C, H, W)`` — pre-processed image pixels.

    Outputs:
        ``logits``: ``(B, num_queries, num_labels + 1)`` — class scores.
        ``pred_boxes``: ``(B, num_queries, 4)`` — sigmoid-normalized (cx, cy, w, h).

    Replicates HuggingFace ``DetrForObjectDetection``.
    """

    default_task: str = "object-detection"
    category: str = "Object Detection"
    config_class: type = DetrConfig

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        num_queries = config.num_queries
        backbone_out_channels = config.backbone_hidden_sizes[-1]

        # Feature map size: ResNet-50 uses stride-32 (2xstem + 4xstage strides)
        image_size = getattr(config, "image_size", 800)
        feat_size = image_size // 32

        # ResNet-50 backbone (no global average pool)
        self.backbone = _DetrResNetBackbone(config)

        # 1x1 projection from backbone output channels to transformer d_model
        self.input_projection = Conv2d(backbone_out_channels, d, kernel_size=1)

        # Fixed sinusoidal spatial position encoding: (1, feat_h*feat_w, d_model)
        pos_data = _compute_sine_pos_embed(feat_size, feat_size, d)
        self.spatial_pos_embed = nn.Parameter(
            list(pos_data.shape),
            data=ir.tensor(pos_data),
        )

        # Learned object query positions: (num_queries, d_model)
        # Added to Q and K in decoder self/cross-attention
        self.query_position_embeddings = nn.Parameter(
            [num_queries, d],
            data=ir.tensor(np.zeros((num_queries, d), dtype=np.float32)),
        )

        # Transformer encoder + decoder
        self.encoder = _DetrEncoder(config)
        self.decoder = _DetrDecoder(config)

        # Detection heads
        self.class_labels_classifier = Linear(d, config.num_labels + 1, bias=True)
        self.bbox_predictor = _MLPPredictionHead(d, d, 4, num_layers=3)

    def forward(
        self, op: builder.OpBuilder, pixel_values: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        d = self.config.d_model
        num_queries = self.config.num_queries

        # ResNet backbone: (B, 3, H, W) → (B, backbone_out, H/32, W/32)
        features = self.backbone(op, pixel_values)

        # Input projection: → (B, d_model, H', W')
        features = self.input_projection(op, features)

        # Flatten spatial dims and permute: (B, d_model, H', W') → (B, H'*W', d_model)
        batch = op.Shape(features, start=0, end=1)  # [B]
        features = op.Transpose(features, perm=[0, 2, 3, 1])  # (B, H', W', d_model)
        features = op.Reshape(
            features,
            op.Concat(batch, op.Constant(value_ints=[-1, d]), axis=0),
        )  # (B, H'*W', d_model)

        # Spatial position encoding (1, H'*W', d_model) — broadcasts over batch
        pos_embed = self.spatial_pos_embed

        # Transformer encoder: (B, H'*W', d_model) → memory (B, H'*W', d_model)
        memory = self.encoder(op, features, pos_embed)

        # Object query positions: (num_queries, d_model) → (1, num_queries, d_model)
        query_pos = op.Unsqueeze(self.query_position_embeddings, [0])

        # Initial decoder target: all-zeros object query slots (B, num_queries, d_model)
        target = op.ConstantOfShape(
            op.Concat(batch, op.Constant(value_ints=[num_queries, d]), axis=0),
            value=ir.tensor(np.zeros(1, dtype=np.float32)),
        )

        # Decoder: (B, num_queries, d_model) — cross-attends to encoder memory
        hs = self.decoder(op, target, memory, query_pos, pos_embed)

        # Class prediction: (B, num_queries, num_labels + 1)
        logits = self.class_labels_classifier(op, hs)

        # Box prediction: (B, num_queries, 4) in [0, 1] after sigmoid
        pred_boxes = op.Sigmoid(self.bbox_predictor(op, hs))

        return logits, pred_boxes

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace DETR weight names to our parameter names.

        Strips the ``model.`` top-level prefix, then maps timm ResNet
        backbone weight naming (``backbone.model.layer{n}.{i}.conv{j}.*``)
        to our ``_DetrResNetBackbone`` convention.

        HuggingFace also stores ``query_position_embeddings`` as a
        ``nn.Embedding`` (key: ``*.weight``); we strip the ``.weight`` suffix
        since our ``nn.Parameter`` has no nested field.
        """
        new: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            # Strip top-level model. wrapper prefix
            if name.startswith("model."):
                name = name[len("model.") :]
            mapped = _rename_detr_weight(name, value)
            if mapped is not None:
                key, val = mapped
                new[key] = val
        return new


# ---------------------------------------------------------------------------
# Weight renaming helper
# ---------------------------------------------------------------------------


def _rename_detr_weight(name: str, value: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Map a single DETR weight name (after stripping ``model.``) to ours.

    Returns ``(new_name, tensor)`` or ``None`` to drop the weight.

    Handles two backbone naming conventions:
    - DETR (facebook/detr-resnet-50):  ``backbone.model.*``
    - TableTransformer:                 ``backbone.conv_encoder.model.*``
    """
    # --- Backbone: timm ResNet naming → our _DetrResNetBackbone naming ---
    # Normalise to a common 'rest' regardless of the exact prefix used
    if name.startswith("backbone.model."):
        rest = name[len("backbone.model.") :]
    elif name.startswith("backbone.conv_encoder.model."):
        rest = name[len("backbone.conv_encoder.model.") :]
    elif name.startswith("backbone."):
        return None  # unrecognised backbone wrapper — drop
    else:
        rest = None

    if rest is not None:
        # Stem: conv1 and bn1
        if rest == "conv1.weight":
            return "backbone.embedder.convolution.weight", value
        if rest.startswith("bn1."):
            return "backbone.embedder.normalization." + rest[len("bn1.") :], value

        # Skip fc/avgpool (not in our backbone)
        if rest.startswith(("fc.", "avgpool")):
            return None

        # Encoder stages: layer{N}.{block}.{component}.{stat}
        m = re.match(r"layer(\d+)\.(\d+)\.(conv\d+|bn\d+|downsample\.\d+)\.(.*)", rest)
        if m:
            stage = int(m.group(1)) - 1  # layer1 → stage 0, layer4 → stage 3
            block = m.group(2)  # block index (0, 1, ...)
            comp = m.group(3)  # conv1, bn2, downsample.0, ...
            stat = m.group(4)  # weight, bias, running_mean, ...
            base = f"backbone.encoder.stages.{stage}.layers.{block}"

            if comp.startswith("conv"):
                j = int(comp[4:]) - 1  # conv1 → 0, conv2 → 1, conv3 → 2
                return f"{base}.layer.{j}.convolution.{stat}", value
            if comp.startswith("bn"):
                j = int(comp[2:]) - 1  # bn1 → 0, bn2 → 1, bn3 → 2
                return f"{base}.layer.{j}.normalization.{stat}", value
            if comp.startswith("downsample."):
                ds = int(comp.split(".")[1])
                sub = "shortcut.convolution" if ds == 0 else "shortcut.normalization"
                return f"{base}.{sub}.{stat}", value

        return None  # unrecognized backbone key — drop it

    # --- query_position_embeddings: nn.Embedding weight → bare parameter ---
    if name == "query_position_embeddings.weight":
        # HF shape: (num_queries, d_model); our Parameter: same shape
        return "query_position_embeddings", value

    # --- Transformer and detection head weights: pass through unchanged ---
    if name.startswith(
        (
            "input_projection.",
            "encoder.",
            "decoder.",
            "class_labels_classifier.",
            "bbox_predictor.",
        )
    ):
        return name, value

    return None  # drop unrecognised keys (e.g. backbone.conv_encoder.*)
