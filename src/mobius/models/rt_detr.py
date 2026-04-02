# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RT-DETR (Real-Time DEtection TRansformer) for object detection.

RT-DETR combines a ResNet backbone with a hybrid encoder (FPN + PAN +
transformer) and a transformer decoder with multi-scale deformable
attention for real-time, end-to-end object detection.

Architecture overview::

    pixel_values (B, 3, H, W)
      → ResNet backbone (multi-scale)
          → S2 (B, 512,  H/8,  W/8)
          → S3 (B, 1024, H/16, W/16)
          → S4 (B, 2048, H/32, W/32)
      → encoder_input_proj: 3x (Conv1x1 + BN)  → all projected to d_model
      → Hybrid encoder:
          AIFI: transformer self-attention on S4
          FPN (top-down): lateral convs + CSP blocks → fuse S4→S3→S2
          PAN (bottom-up): downsample convs + CSP blocks → fuse S2→S3→S4
      → decoder_input_proj: 3x (Conv1x1 + BN)
      → Flatten + anchor generation + TopK query selection
      → Transformer decoder (6 layers):
          self-attention + multi-scale deformable cross-attention + FFN
          per-layer box refinement (class_embed + bbox_embed)
      → logits (B, num_queries, num_labels)
      → pred_boxes (B, num_queries, 4)

Multi-scale deformable attention samples values from multi-scale feature
maps at learned offsets around reference points, using ONNX GridSample.

Replicates HuggingFace ``RTDetrForObjectDetection``.
"""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import RtDetrConfig
from mobius.components import LayerNorm, Linear
from mobius.components._conv import BatchNorm2d, Conv2dNoBias

# ---------------------------------------------------------------------------
# Shared CNN building blocks (Conv + BN + activation, RepVgg, CSP)
# ---------------------------------------------------------------------------


class _ConvNorm(nn.Module):
    """Conv2d → BatchNorm2d → optional activation (SiLU by default)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        activate: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = Conv2dNoBias(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = BatchNorm2d(out_channels)
        self._activate = activate

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.conv(op, x)
        x = self.norm(op, x)
        if self._activate:
            x = op.Mul(x, op.Sigmoid(x))  # SiLU = x * sigmoid(x)
        return x


class _RepVggBlock(nn.Module):
    """RepVGG block: parallel 3x3 and 1x1 convolutions added then activated.

    Structure: ``SiLU(Conv3x3_BN(x) + Conv1x1_BN(x))``
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _ConvNorm(channels, channels, kernel_size=3, padding=1, activate=False)
        self.conv2 = _ConvNorm(channels, channels, kernel_size=1, padding=0, activate=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        y = op.Add(self.conv1(op, x), self.conv2(op, x))
        return op.Mul(y, op.Sigmoid(y))  # SiLU


class _CSPRepLayer(nn.Module):
    """CSP block with RepVGG bottlenecks (used in FPN/PAN stages).

    Structure::

        conv1(x) → bottleneck chain → +
        conv2(x) ────────────────────→+→ conv3
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = _ConvNorm(in_channels, hidden, kernel_size=1)
        self.conv2 = _ConvNorm(in_channels, hidden, kernel_size=1)
        self.bottlenecks = nn.Sequential(*[_RepVggBlock(hidden) for _ in range(num_blocks)])
        # Identity when hidden == out_channels (common case)
        self._use_conv3 = hidden != out_channels
        if self._use_conv3:
            self.conv3 = _ConvNorm(hidden, out_channels, kernel_size=1)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        branch1 = self.bottlenecks(op, self.conv1(op, x))
        branch2 = self.conv2(op, x)
        out = op.Add(branch1, branch2)
        if self._use_conv3:
            out = self.conv3(op, out)
        return out


# ---------------------------------------------------------------------------
# ResNet backbone with multi-scale output (RT-DETR variant)
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Module):
    """Conv2dNoBias → BatchNorm2d → ReLU. Used in backbone stem and blocks."""

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


class _RtDetrShortcut(nn.Module):
    """Shortcut with optional AvgPool for stride-2 downsampling.

    RT-DETR ResNet uses AvgPool2d + 1x1 Conv for stride-2 shortcuts
    (instead of strided convolution used in standard ResNet).

    Weight names:
        - stride==1: ``shortcut.convolution.*`` / ``shortcut.normalization.*``
        - stride==2: ``shortcut.1.convolution.*`` / ``shortcut.1.normalization.*``
            (index 0 is AvgPool which has no weights)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self._stride = stride
        self.convolution = Conv2dNoBias(in_channels, out_channels, kernel_size=1, stride=1)
        self.normalization = BatchNorm2d(out_channels)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        if self._stride > 1:
            x = op.AveragePool(
                x,
                kernel_shape=[self._stride, self._stride],
                strides=[self._stride, self._stride],
            )
        x = self.convolution(op, x)
        x = self.normalization(op, x)
        return x


class _BottleneckBlock(nn.Module):
    """Bottleneck block: 1x1 → 3x3 → 1x1 + shortcut.

    Stride-2 downsampling is in the 3x3 conv (layer.1).
    Shortcut uses AvgPool + Conv (not strided conv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 4,
    ):
        super().__init__()
        mid = out_channels // reduction
        self.layer = nn.ModuleList(
            [
                _ConvBnRelu(in_channels, mid, kernel_size=1),
                _ConvBnRelu(mid, mid, kernel_size=3, stride=stride, padding=1),
                _ConvBnRelu(mid, out_channels, kernel_size=1, activate=False),
            ]
        )
        self._use_shortcut = (in_channels != out_channels) or stride != 1
        if self._use_shortcut:
            self.shortcut = _RtDetrShortcut(in_channels, out_channels, stride)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        residual = x
        for conv in self.layer:
            x = conv(op, x)
        if self._use_shortcut:
            residual = self.shortcut(op, residual)
        return op.Relu(op.Add(x, residual))


class _ResNetStage(nn.Module):
    """One stage of bottleneck blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            s = stride if i == 0 else 1
            ic = in_channels if i == 0 else out_channels
            blocks.append(_BottleneckBlock(ic, out_channels, stride=s))
        self.layers = nn.Sequential(*blocks)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.layers(op, x)


class _RtDetrResNetBackbone(nn.Module):
    """RT-DETR ResNet backbone with multi-scale output.

    The stem uses 3x Conv3x3 (unlike standard ResNet's single 7x7 conv).
    Returns feature maps from the requested stages (typically stages 1,2,3
    for the 3-level FPN).
    """

    def __init__(self, config: RtDetrConfig):
        super().__init__()
        embed = config.backbone_embedding_size
        hidden_sizes = config.backbone_hidden_sizes
        depths = config.backbone_depths
        num_channels = getattr(config, "num_channels", 3)

        # Stem: 3x Conv3x3 with BN + ReLU
        self.embedder = nn.ModuleList(
            [
                _ConvBnRelu(num_channels, embed // 2, 3, stride=2, padding=1),
                _ConvBnRelu(embed // 2, embed // 2, 3, stride=1, padding=1),
                _ConvBnRelu(embed // 2, embed, 3, stride=1, padding=1),
            ]
        )

        # 4 stages of bottleneck blocks
        in_ch = [embed, *list(hidden_sizes[:-1])]
        self.stages = nn.ModuleList(
            [
                _ResNetStage(
                    in_ch[i],
                    hidden_sizes[i],
                    depths[i],
                    stride=1 if i == 0 else 2,
                )
                for i in range(4)
            ]
        )

        self._out_indices = list(config.backbone_out_indices)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> list[ir.Value]:
        # Stem: (B, 3, H, W) → (B, embed, H/2, W/2)
        x = pixel_values
        for conv in self.embedder:
            x = conv(op, x)
        # MaxPool: → (B, embed, H/4, W/4)
        x = op.MaxPool(x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

        # Stages: collect multi-scale features at requested out_indices
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(op, x)
            if i in self._out_indices:
                features.append(x)
        return features


# ---------------------------------------------------------------------------
# 2D sinusoidal position encoding (computed at graph construction time)
# ---------------------------------------------------------------------------


def _compute_sine_pos_embed_2d(
    h: int, w: int, d_model: int, temperature: float = 10000.0
) -> np.ndarray:
    """2D sine/cosine positional encoding for RT-DETR AIFI layer.

    Formula matches HuggingFace ``RTDetrSinePositionEmbedding``:
    pos_dim = d_model // 4, then for each grid position:
    ``[sin(h*ω), cos(h*ω), sin(w*ω), cos(w*ω)]`` concatenated.

    Returns: shape ``(1, h*w, d_model)`` float32.
    """
    assert d_model % 4 == 0
    pos_dim = d_model // 4
    omega = np.arange(pos_dim, dtype=np.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    grid_w = np.arange(w, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    # meshgrid with "xy" indexing: grid_w varies across columns
    gw, gh = np.meshgrid(grid_w, grid_h, indexing="xy")

    out_w = gw.flatten()[:, None] @ omega[None, :]  # (h*w, pos_dim)
    out_h = gh.flatten()[:, None] @ omega[None, :]  # (h*w, pos_dim)

    pos = np.concatenate(
        [np.sin(out_h), np.cos(out_h), np.sin(out_w), np.cos(out_w)],
        axis=1,
    )  # (h*w, d_model)
    return pos.reshape(1, h * w, d_model).astype(np.float32)


# ---------------------------------------------------------------------------
# AIFI encoder layer (standard transformer encoder with sine pos embed)
# ---------------------------------------------------------------------------


class _AIFIEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (matches RTDetrSelfAttention layer).

    Structure::

        residual + self_attn(x + pos_embed) → LayerNorm
        residual + mlp(x)                   → LayerNorm
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.o_proj = Linear(d_model, d_model, bias=True)
        self.self_attn_layer_norm = LayerNorm(d_model, eps=1e-5)
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model, eps=1e-5)

    def forward(self, op: builder.OpBuilder, x: ir.Value, pos_embed: ir.Value) -> ir.Value:
        # Self-attention with positional bias on Q and K
        residual = x
        q_input = op.Add(x, pos_embed)
        k_input = op.Add(x, pos_embed)

        q = self.q_proj(op, q_input)
        k = self.k_proj(op, k_input)
        v = self.v_proj(op, x)

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
        )
        attn_out = self.o_proj(op, attn_out)
        x = self.self_attn_layer_norm(op, op.Add(residual, attn_out))

        # FFN with ReLU activation
        residual = x
        ffn_out = self.fc2(op, op.Relu(self.fc1(op, x)))
        x = self.final_layer_norm(op, op.Add(residual, ffn_out))
        return x


# ---------------------------------------------------------------------------
# Hybrid encoder: AIFI + FPN (top-down) + PAN (bottom-up)
# ---------------------------------------------------------------------------


class _HybridEncoder(nn.Module):
    """RT-DETR hybrid encoder: AIFI self-attention + CCFM neck.

    1. AIFI: apply transformer self-attention on the highest-level feature
    2. FPN (top-down): fuse high→low with lateral convs + CSP blocks
    3. PAN (bottom-up): fuse low→high with downsample convs + CSP blocks
    """

    def __init__(self, config: RtDetrConfig):
        super().__init__()
        d = config.d_model
        n_levels = config.num_feature_levels  # typically 3

        # Compute spatial shape for highest-level feature (AIFI input)
        image_size = getattr(config, "image_size", 640)
        highest_stride = config.feat_strides[-1]
        aifi_h = image_size // highest_stride
        aifi_w = image_size // highest_stride

        # AIFI: transformer layers on highest feature level
        self.aifi = nn.ModuleList(
            [
                _AIFIWrapper(
                    d,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    config.encoder_layers,
                    config.positional_encoding_temperature,
                    spatial_shape=(aifi_h, aifi_w),
                )
            ]
        )

        # FPN (top-down): n_levels - 1 lateral convs + CSP blocks
        self.lateral_convs = nn.ModuleList(
            [_ConvNorm(d, d, kernel_size=1) for _ in range(n_levels - 1)]
        )
        self.fpn_blocks = nn.ModuleList([_CSPRepLayer(d * 2, d) for _ in range(n_levels - 1)])

        # PAN (bottom-up): n_levels - 1 downsample convs + CSP blocks
        self.downsample_convs = nn.ModuleList(
            [_ConvNorm(d, d, kernel_size=3, stride=2, padding=1) for _ in range(n_levels - 1)]
        )
        self.pan_blocks = nn.ModuleList([_CSPRepLayer(d * 2, d) for _ in range(n_levels - 1)])

    def forward(
        self,
        op: builder.OpBuilder,
        features: list[ir.Value],
    ) -> list[ir.Value]:
        # AIFI: transformer on highest-level feature (last in list)
        features[-1] = self.aifi[0](op, features[-1])

        # Top-down FPN: process from highest to lowest resolution
        fpn = [features[-1]]
        num_fpn = len(self.lateral_convs)
        for idx in range(num_fpn):
            backbone_feat = features[num_fpn - idx - 1]
            top_feat = self.lateral_convs[idx](op, fpn[-1])
            fpn[-1] = top_feat

            # Upsample 2x using nearest-neighbor interpolation
            target_h = op.Shape(backbone_feat, start=2, end=3)
            target_w = op.Shape(backbone_feat, start=3, end=4)
            upsampled = op.Resize(
                top_feat,
                None,  # roi
                None,  # scales
                op.Concat(
                    op.Shape(top_feat, start=0, end=2),
                    target_h,
                    target_w,
                    axis=0,
                ),
                mode="nearest",
            )

            fused = op.Concat(upsampled, backbone_feat, axis=1)
            fpn.append(self.fpn_blocks[idx](op, fused))

        fpn.reverse()  # now low-to-high resolution order

        # Bottom-up PAN
        pan = [fpn[0]]
        for idx in range(num_fpn):
            top_pan = self.downsample_convs[idx](op, pan[-1])
            fused = op.Concat(top_pan, fpn[idx + 1], axis=1)
            pan.append(self.pan_blocks[idx](op, fused))

        return pan


class _AIFIWrapper(nn.Module):
    """Wraps the AIFI transformer layer with flatten/unflatten and pos embed.

    Handles: 4D feature → flatten → add pos → transformer → reshape to 4D.
    The position embedding is pre-computed as a constant since spatial
    dimensions are determined by image_size and feat_strides at build time.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        temperature: float = 10000.0,
        spatial_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        self._d_model = d_model
        self.layers = nn.ModuleList(
            [_AIFIEncoderLayer(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        # Pre-compute sine position encoding as a constant initializer
        if spatial_shape is not None:
            h, w = spatial_shape
            pos_data = _compute_sine_pos_embed_2d(h, w, d_model, temperature)
            self._pos_embed = nn.Parameter(
                list(pos_data.shape),
                data=ir.tensor(pos_data),
            )
            self._h = h
            self._w = w
        else:
            self._pos_embed = None
            self._h = 0
            self._w = 0

    def forward(self, op: builder.OpBuilder, feature: ir.Value) -> ir.Value:
        # feature: (B, C, H, W)
        batch = op.Shape(feature, start=0, end=1)

        # Flatten: (B, C, H, W) → (B, H*W, C)
        x = op.Transpose(feature, perm=[0, 2, 3, 1])  # (B, H, W, C)
        x = op.Reshape(
            x,
            op.Concat(batch, op.Constant(value_ints=[-1, self._d_model]), axis=0),
        )

        # Use pre-computed position embedding
        pos = self._pos_embed

        for layer in self.layers:
            x = layer(op, x, pos)

        # Unflatten: (B, H*W, C) → (B, C, H, W)
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._h, self._w, self._d_model]),
                axis=0,
            ),
        )
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x


# ---------------------------------------------------------------------------
# Multi-scale deformable attention (decoder cross-attention)
# ---------------------------------------------------------------------------


class _DeformableAttention(nn.Module):
    """Multi-scale deformable attention for RT-DETR decoder.

    For each query, predicts sampling offsets and attention weights to sample
    from multi-scale encoder feature maps using bilinear interpolation
    (ONNX GridSample op).

    Parameters:
        - sampling_offsets: (d_model → n_heads * n_levels * n_points * 2)
        - attention_weights: (d_model → n_heads * n_levels * n_points)
        - value_proj: (d_model → d_model)
        - output_proj: (d_model → d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
    ):
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_levels = n_levels
        self._n_points = n_points
        self._d_head = d_model // n_heads

        self.sampling_offsets = Linear(d_model, n_heads * n_levels * n_points * 2, bias=True)
        self.attention_weights = Linear(d_model, n_heads * n_levels * n_points, bias=True)
        self.value_proj = Linear(d_model, d_model, bias=True)
        self.output_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> ir.Value:
        """Multi-scale deformable cross-attention.

        Args:
            hidden_states: (B, Q, D) decoder query features
            encoder_hidden_states: (B, S, D) flattened multi-scale features
            reference_points: (B, Q, 1, 4) in [0,1], [x, y, w, h]
            spatial_shapes: list of (H, W) per level (known at build time)
            level_start_index: list of start indices per level
        """
        n_heads = self._n_heads
        n_levels = self._n_levels
        n_points = self._n_points
        d_head = self._d_head

        # Project values: (B, S, D) → (B, S, n_heads, d_head)
        value = self.value_proj(op, encoder_hidden_states)
        value = op.Reshape(
            value,
            op.Concat(
                op.Shape(value, start=0, end=2),
                op.Constant(value_ints=[n_heads, d_head]),
                axis=0,
            ),
        )

        # Predict offsets: (B, Q, H*L*P*2) → (B, Q, H, L, P, 2)
        offsets = self.sampling_offsets(op, hidden_states)
        offsets = op.Reshape(
            offsets,
            op.Concat(
                op.Shape(offsets, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels, n_points, 2]),
                axis=0,
            ),
        )

        # Predict attention weights: (B, Q, H*L*P) → softmax → (B, Q, H, L, P)
        attn_w = self.attention_weights(op, hidden_states)
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Shape(attn_w, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels * n_points]),
                axis=0,
            ),
        )
        attn_w = op.Softmax(attn_w, axis=-1)
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Shape(attn_w, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels, n_points]),
                axis=0,
            ),
        )

        # Compute sampling locations from reference_points + offsets
        # reference_points: (B, Q, 1, 4) → split into xy and wh
        ref_xy = op.Slice(
            reference_points,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[2]),
            op.Constant(value_ints=[3]),
        )  # (B, Q, 1, 2)
        ref_wh = op.Slice(
            reference_points,
            op.Constant(value_ints=[2]),
            op.Constant(value_ints=[4]),
            op.Constant(value_ints=[3]),
        )  # (B, Q, 1, 2)

        # Expand ref for broadcasting: (B, Q, 1, 1, 1, 2)
        ref_xy = op.Unsqueeze(op.Unsqueeze(ref_xy, [2]), [4])
        ref_wh = op.Unsqueeze(op.Unsqueeze(ref_wh, [2]), [4])

        # sampling_locations = ref_xy + offsets / n_points * ref_wh * 0.5
        normalized_offsets = op.Mul(
            op.Div(offsets, op.Constant(value_float=float(n_points))),
            op.Mul(ref_wh, op.Constant(value_float=0.5)),
        )
        sampling_locs = op.Add(ref_xy, normalized_offsets)
        # → (B, Q, H, L, P, 2)

        # Convert to grid_sample coordinates: grid = 2 * loc - 1
        sampling_grids = op.Sub(
            op.Mul(sampling_locs, op.Constant(value_float=2.0)),
            op.Constant(value_float=1.0),
        )

        # For each level: extract value slice, grid_sample, collect
        sampled_values = []
        for level_id, (h_l, w_l) in enumerate(spatial_shapes):
            start = level_start_index[level_id]
            length = h_l * w_l

            # value_l: (B, h_l*w_l, H, d_head)
            value_l = op.Slice(
                value,
                op.Constant(value_ints=[start]),
                op.Constant(value_ints=[start + length]),
                op.Constant(value_ints=[1]),
            )
            # → (B, H, d_head, h_l*w_l) → (B*H, d_head, h_l, w_l)
            value_l = op.Transpose(value_l, perm=[0, 2, 3, 1])
            batch = op.Shape(value_l, start=0, end=1)
            value_l = op.Reshape(
                value_l,
                op.Concat(
                    op.Mul(batch, op.Constant(value_ints=[n_heads])),
                    op.Constant(value_ints=[d_head, h_l, w_l]),
                    axis=0,
                ),
            )

            # grid_l: extract level_id from dim 3
            # sampling_grids: (B, Q, H, L, P, 2) → take [:,:,:,level_id,:,:]
            grid_l = op.Gather(
                sampling_grids, op.Constant(value_int=level_id), axis=3
            )  # (B, Q, H, P, 2)
            # → (B, H, Q, P, 2) → (B*H, Q, P, 2)
            grid_l = op.Transpose(grid_l, perm=[0, 2, 1, 3, 4])
            grid_l = op.Reshape(
                grid_l,
                op.Concat(
                    op.Mul(batch, op.Constant(value_ints=[n_heads])),
                    op.Shape(grid_l, start=2, end=5),
                    axis=0,
                ),
            )

            # GridSample: (B*H, d_head, h_l, w_l) + (B*H, Q, P, 2)
            #           → (B*H, d_head, Q, P)
            sampled_l = op.GridSample(
                value_l,
                grid_l,
                align_corners=0,
                mode="bilinear",
                padding_mode="zeros",
            )
            sampled_values.append(sampled_l)

        # Stack along level dimension and flatten: (B*H, d_head, Q, L*P)
        output = op.Concat(*sampled_values, axis=-1)

        # Apply attention weights
        # attn_w: (B, Q, H, L, P) → (B, H, Q, L*P) → (B*H, 1, Q, L*P)
        attn_w = op.Transpose(attn_w, perm=[0, 2, 1, 3, 4])
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Mul(batch, op.Constant(value_ints=[n_heads])),
                op.Constant(value_ints=[1, -1, n_levels * n_points]),
                axis=0,
            ),
        )

        # Weighted sum: (B*H, d_head, Q, L*P) * (B*H, 1, Q, L*P)
        # → sum over L*P → (B*H, d_head, Q)
        output = op.ReduceSum(
            op.Mul(output, attn_w),
            op.Constant(value_ints=[-1]),
            keepdims=0,
        )

        # Reshape: (B*H, d_head, Q) → (B, H*d_head, Q) → (B, Q, D)
        output = op.Reshape(
            output,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._d_model, -1]),
                axis=0,
            ),
        )
        output = op.Transpose(output, perm=[0, 2, 1])

        return self.output_proj(op, output)


# ---------------------------------------------------------------------------
# Decoder layer and decoder stack
# ---------------------------------------------------------------------------


class _RtDetrDecoderLayer(nn.Module):
    """RT-DETR decoder layer: self-attn + deformable cross-attn + FFN.

    All sub-layers use post-norm (residual → add → LayerNorm).
    """

    def __init__(self, config: RtDetrConfig):
        super().__init__()
        d = config.d_model

        # Self-attention (standard multi-head)
        self.self_attn = _SelfAttention(d, config.decoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(d, eps=1e-5)

        # Cross-attention (multi-scale deformable)
        self.encoder_attn = _DeformableAttention(
            d,
            config.decoder_attention_heads,
            config.num_feature_levels,
            config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = LayerNorm(d, eps=1e-5)

        # FFN
        self.mlp = _FFN(d, config.decoder_ffn_dim)
        self.final_layer_norm = LayerNorm(d, eps=1e-5)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        query_pos: ir.Value,
        encoder_hidden_states: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> ir.Value:
        # Self-attention: queries attend to each other with position embedding
        residual = hidden_states
        q = op.Add(hidden_states, query_pos)
        k = op.Add(hidden_states, query_pos)
        attn_out = self.self_attn(op, q, k, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, attn_out))

        # Deformable cross-attention with position embedding added to query
        residual = hidden_states
        cross_input = op.Add(hidden_states, query_pos)
        cross_out = self.encoder_attn(
            op,
            cross_input,
            encoder_hidden_states,
            reference_points,
            spatial_shapes,
            level_start_index,
        )
        hidden_states = self.encoder_attn_layer_norm(op, op.Add(residual, cross_out))

        # FFN
        residual = hidden_states
        ffn_out = self.mlp(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, ffn_out))

        return hidden_states


class _SelfAttention(nn.Module):
    """Standard multi-head self-attention (used in decoder self-attn)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.o_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
    ) -> ir.Value:
        q = self.q_proj(op, query)
        k = self.k_proj(op, key)
        v = self.v_proj(op, value)
        attn = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
        )
        return self.o_proj(op, attn)


class _FFN(nn.Module):
    """Two-layer FFN with ReLU. Named fc1/fc2 to match HF weight names."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.fc2(op, op.Relu(self.fc1(op, x)))


class _MLPHead(nn.Module):
    """MLP prediction head (used for bbox regression and query_pos_head).

    N Linear layers with ReLU between (no activation after last layer).
    """

    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )
        self._num_layers = len(dims) - 1

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for i, layer in enumerate(self.layers):
            x = layer(op, x)
            if i < self._num_layers - 1:
                x = op.Relu(x)
        return x


# ---------------------------------------------------------------------------
# Anchor generation (computed at graph construction time as constants)
# ---------------------------------------------------------------------------


def _generate_anchors(
    spatial_shapes: list[tuple[int, int]],
    grid_size: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute anchor boxes and validity mask.

    Matches HuggingFace ``RTDetrModel.generate_anchors``.

    Returns:
        anchors: (1, total_hw, 4) float32 — log-space anchor coordinates.
        valid_mask: (1, total_hw, 1) float32 — 1.0 where anchor is valid.
    """
    all_anchors = []
    for level, (h, w) in enumerate(spatial_shapes):
        grid_y, grid_x = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        grid_xy = np.stack([grid_x, grid_y], axis=-1).reshape(1, h * w, 2)
        grid_xy = grid_xy + 0.5
        grid_xy[..., 0] /= w
        grid_xy[..., 1] /= h
        wh = np.ones_like(grid_xy) * grid_size * (2.0**level)
        all_anchors.append(
            np.concatenate([grid_xy, wh], axis=-1)  # (1, h*w, 4)
        )

    anchors = np.concatenate(all_anchors, axis=1)  # (1, total, 4)
    eps = 1e-2
    valid = ((anchors > eps) & (anchors < 1 - eps)).all(axis=-1, keepdims=True)
    valid_mask = valid.astype(np.float32)

    # Convert to log-space; invalid positions get large values
    anchors_clipped = np.clip(anchors, 1e-6, 1 - 1e-6)
    log_anchors = np.log(anchors_clipped / (1 - anchors_clipped))
    log_anchors = np.where(valid, log_anchors, np.float32(1e8))

    return log_anchors.astype(np.float32), valid_mask


def _inverse_sigmoid(op: builder.OpBuilder, x: ir.Value) -> ir.Value:
    """Inverse sigmoid: log(x / (1 - x)) with clamping for stability."""
    eps = 1e-5
    x1 = op.Clip(x, op.Constant(value_float=eps))
    x2 = op.Clip(
        op.Sub(op.Constant(value_float=1.0), x),
        op.Constant(value_float=eps),
    )
    return op.Log(op.Div(x1, x2))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class RtDetrForObjectDetection(nn.Module):
    """RT-DETR (Real-Time DEtection TRansformer) for object detection.

    Combines a ResNet backbone with a hybrid encoder (FPN/PAN/AIFI)
    and a transformer decoder with deformable cross-attention for
    real-time end-to-end object detection.

    Inputs:
        ``pixel_values``: ``(B, C, H, W)`` — preprocessed image.

    Outputs:
        ``logits``: ``(B, num_queries, num_labels)`` — class scores.
        ``pred_boxes``: ``(B, num_queries, 4)`` — normalized (cx, cy, w, h).

    Replicates HuggingFace ``RTDetrForObjectDetection``.
    """

    default_task: str = "object-detection"
    category: str = "Object Detection"
    config_class: type = RtDetrConfig

    def __init__(self, config: RtDetrConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        n_levels = config.num_feature_levels

        # ResNet backbone → multi-scale features
        self.backbone = _RtDetrResNetBackbone(config)

        # Input projections: backbone channels → d_model
        self.encoder_input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dNoBias(ch, d, kernel_size=1),
                    BatchNorm2d(d),
                )
                for ch in config.encoder_in_channels
            ]
        )

        # Hybrid encoder
        self.encoder = _HybridEncoder(config)

        # Decoder input projections
        self.decoder_input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dNoBias(d, d, kernel_size=1),
                    BatchNorm2d(d),
                )
                for _ in range(n_levels)
            ]
        )

        # Encoder output head (for query selection)
        self.enc_output = nn.Sequential(
            Linear(d, d, bias=True),
            LayerNorm(d, eps=1e-5),
        )
        self.enc_score_head = Linear(d, config.num_labels, bias=True)
        self.enc_bbox_head = _MLPHead([d, d, d, 4])

        # Decoder
        self.decoder = _RtDetrDecoder(config)

        # Pre-compute spatial shapes and anchors for the fixed image size
        image_size = getattr(config, "image_size", 640)
        self._spatial_shapes = [
            (image_size // s, image_size // s) for s in config.feat_strides
        ]
        self._level_start_index = []
        idx = 0
        for h, w in self._spatial_shapes:
            self._level_start_index.append(idx)
            idx += h * w

        # Pre-compute anchors as constant initializers
        anchors_np, valid_mask_np = _generate_anchors(self._spatial_shapes)

        self._anchors = nn.Parameter(
            list(anchors_np.shape),
            data=ir.tensor(anchors_np),
        )
        self._valid_mask = nn.Parameter(
            list(valid_mask_np.shape),
            data=ir.tensor(valid_mask_np),
        )

    def forward(
        self, op: builder.OpBuilder, pixel_values: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        d = self.config.d_model
        num_queries = self.config.num_queries

        # 1. Backbone: extract multi-scale features
        features = self.backbone(op, pixel_values)

        # 2. Project backbone features to d_model
        proj_feats = []
        for i, feat in enumerate(features):
            proj_feats.append(self.encoder_input_proj[i](op, feat))

        # 3. Hybrid encoder: AIFI + FPN + PAN
        encoder_outputs = self.encoder(op, proj_feats)

        # 4. Decoder input projections
        sources = []
        for i, src in enumerate(encoder_outputs):
            sources.append(self.decoder_input_proj[i](op, src))

        # 5. Flatten multi-scale features for decoder
        # Each source: (B, d_model, H_l, W_l) → (B, H_l*W_l, d_model)
        source_flatten_parts = []
        batch = op.Shape(sources[0], start=0, end=1)
        for src in sources:
            s = op.Transpose(src, perm=[0, 2, 3, 1])  # (B, H, W, D)
            s = op.Reshape(
                s,
                op.Concat(batch, op.Constant(value_ints=[-1, d]), axis=0),
            )
            source_flatten_parts.append(s)
        source_flatten = op.Concat(*source_flatten_parts, axis=1)

        # 6. Apply valid mask and encode
        memory = op.Mul(self._valid_mask, source_flatten)
        output_memory = self.enc_output(op, memory)

        # 7. Score and select top-k queries
        enc_scores = self.enc_score_head(op, output_memory)
        enc_bboxes_logits = op.Add(
            self.enc_bbox_head(op, output_memory),
            self._anchors,
        )

        # TopK on max class score per position
        max_scores = op.ReduceMax(enc_scores, op.Constant(value_ints=[-1]), keepdims=0)
        _topk_vals, topk_ind = op.TopK(
            max_scores,
            op.Constant(value_int=num_queries),
            axis=1,
            _outputs=2,
        )  # topk_ind: (B, num_queries)

        # Gather reference points and initial target
        # Expand topk_ind for gather: (B, Q, 1) → (B, Q, 4) for bboxes
        topk_ind_4 = op.Expand(
            op.Unsqueeze(topk_ind, [2]),
            op.Concat(
                batch,
                op.Constant(value_ints=[num_queries, 4]),
                axis=0,
            ),
        )
        reference_points_unact = op.GatherElements(enc_bboxes_logits, topk_ind_4, axis=1)

        # Expand for d_model gather
        topk_ind_d = op.Expand(
            op.Unsqueeze(topk_ind, [2]),
            op.Concat(
                batch,
                op.Constant(value_ints=[num_queries, d]),
                axis=0,
            ),
        )
        target = op.GatherElements(output_memory, topk_ind_d, axis=1)

        # 8. Decoder with iterative box refinement
        logits, pred_boxes = self.decoder(
            op,
            target,
            source_flatten,
            reference_points_unact,
            self._spatial_shapes,
            self._level_start_index,
        )

        # Explicit shape annotation for shape inference
        batch_sym = ir.SymbolicDim("batch")
        logits.shape = ir.Shape([batch_sym, num_queries, self.config.num_labels])
        logits.type = ir.TensorType(ir.DataType.FLOAT)
        pred_boxes.shape = ir.Shape([batch_sym, num_queries, 4])
        pred_boxes.type = ir.TensorType(ir.DataType.FLOAT)

        return logits, pred_boxes

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace RT-DETR weights to our parameter names."""
        new: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            # Strip ForObjectDetection model. prefix
            if name.startswith("model."):
                name = name[len("model.") :]
            result = _rename_rt_detr_weight(name, value)
            if result is not None:
                new[result[0]] = result[1]
        return new


class _RtDetrDecoder(nn.Module):
    """RT-DETR decoder with iterative box refinement.

    Each layer refines bounding boxes and produces per-layer class logits.
    The final output uses the last layer's predictions.
    """

    def __init__(self, config: RtDetrConfig):
        super().__init__()
        d = config.d_model
        n = config.decoder_layers

        # Position encoding from reference points: 4 → d_model
        self.query_pos_head = _MLPHead([4, d * 2, d])

        # Decoder layers
        self.layers = nn.ModuleList([_RtDetrDecoderLayer(config) for _ in range(n)])

        # Per-layer detection heads (iterative refinement)
        self.class_embed = nn.ModuleList(
            [Linear(d, config.num_labels, bias=True) for _ in range(n)]
        )
        self.bbox_embed = nn.ModuleList([_MLPHead([d, d, d, 4]) for _ in range(n)])

    def forward(
        self,
        op: builder.OpBuilder,
        target: ir.Value,
        encoder_hidden_states: ir.Value,
        reference_points_unact: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> tuple[ir.Value, ir.Value]:
        """Run decoder layers with iterative box refinement.

        Returns logits and pred_boxes from the LAST decoder layer.
        """
        hidden_states = target
        # Sigmoid of initial reference points
        reference_points = op.Sigmoid(reference_points_unact)

        for idx, layer in enumerate(self.layers):
            # Generate position embeddings from current reference points
            query_pos = self.query_pos_head(op, reference_points)

            # reference_points_input: (B, Q, 4) → (B, Q, 1, 4) for deformable attn
            ref_input = op.Unsqueeze(reference_points, [2])

            hidden_states = layer(
                op,
                hidden_states,
                query_pos,
                encoder_hidden_states,
                ref_input,
                spatial_shapes,
                level_start_index,
            )

            # Box refinement: predict delta + add to inverse-sigmoid of ref
            predicted_corners = self.bbox_embed[idx](op, hidden_states)
            new_ref = op.Sigmoid(
                op.Add(
                    predicted_corners,
                    _inverse_sigmoid(op, reference_points),
                )
            )
            reference_points = new_ref

        # Final layer class prediction
        logits = self.class_embed[-1](op, hidden_states)
        pred_boxes = reference_points

        return logits, pred_boxes


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------

# Regex for backbone encoder stage weights
_BACKBONE_STAGE_RE = re.compile(r"backbone\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(.*)")


def _rename_rt_detr_weight(name: str, value: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Map a single HF RT-DETR weight name to our naming convention.

    All names have already had the ``model.`` prefix stripped.
    """
    # Skip training-only weights
    if "num_batches_tracked" in name:
        return None
    if name.startswith("denoising_class_embed"):
        return None

    # --- Backbone ---
    if name.startswith("backbone.model."):
        rest = name[len("backbone.model.") :]

        # Stem: embedder.embedder.{0,1,2} → backbone.embedder.{0,1,2}
        if rest.startswith("embedder.embedder."):
            suffix = rest[len("embedder.embedder.") :]
            return f"backbone.embedder.{suffix}", value

        # Encoder stages
        if rest.startswith("encoder."):
            suffix = rest[len("encoder.") :]
            new_name = f"backbone.encoder.{suffix}"
            # Handle stride-2 shortcut naming:
            # HF: shortcut.1.convolution → our: shortcut.convolution
            new_name = re.sub(
                r"\.shortcut\.1\.(convolution|normalization)",
                r".shortcut.\1",
                new_name,
            )
            return new_name, value

        return None

    # --- Encoder input projections: pass through ---
    if name.startswith("encoder_input_proj."):
        return name, value

    # --- Encoder (AIFI + FPN/PAN) ---
    if name.startswith("encoder."):
        # AIFI: encoder.aifi.0.layers.0.self_attn.* →
        #        encoder.aifi.0.layers.0.* (flatten 'self_attn.' for
        #        the q/k/v/o_proj which are direct children)
        new = name
        # The AIFI layer has self_attn.{q,k,v,o}_proj at HF level
        # but our _AIFIEncoderLayer has q/k/v/o_proj directly
        new = re.sub(
            r"encoder\.aifi\.0\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
            r"encoder.aifi.0.layers.\1.\2",
            new,
        )
        # The mlp.fc1/fc2 → just fc1/fc2
        new = re.sub(
            r"encoder\.aifi\.0\.layers\.(\d+)\.mlp\.(fc1|fc2)",
            r"encoder.aifi.0.layers.\1.\2",
            new,
        )
        return new, value

    # --- Decoder input projections: pass through ---
    if name.startswith("decoder_input_proj."):
        return name, value

    # --- Encoder output heads: pass through ---
    if name.startswith(("enc_output.", "enc_score_head.", "enc_bbox_head.")):
        return name, value

    # --- Decoder ---
    if name.startswith("decoder."):
        return name, value

    return None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = ["RtDetrForObjectDetection"]
