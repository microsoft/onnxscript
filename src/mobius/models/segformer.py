# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Segformer hierarchical vision transformer for semantic segmentation.

Segformer uses a multi-stage encoder with overlapping patch embeddings,
efficient self-attention with sequence reduction, and Mix-FFN with
depthwise convolutions. The decode head fuses multi-scale features
through linear projections and upsampling.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import SegformerConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import BatchNorm2d, Conv2d, Conv2dNoBias

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------


class _OverlapPatchEmbeddings(nn.Module):
    """Overlapping patch embedding: Conv2d with padding → flatten → LayerNorm."""

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        embeddings = self.proj(op, pixel_values)
        # Save spatial dims for later reshape
        height = op.Shape(embeddings, start=2, end=3)
        width = op.Shape(embeddings, start=3, end=4)
        # [B, C, H, W] → [B, H*W, C]
        batch = op.Shape(embeddings, start=0, end=1)
        channels = op.Shape(embeddings, start=1, end=2)
        embeddings = op.Reshape(
            embeddings, op.Concat(batch, channels, op.Constant(value_ints=[-1]), axis=0)
        )
        embeddings = op.Transpose(embeddings, perm=[0, 2, 1])
        embeddings = self.layer_norm(op, embeddings)
        return embeddings, height, width


class _EfficientSelfAttention(nn.Module):
    """Efficient self-attention with optional sequence reduction.

    When sr_ratio > 1, K and V sequences are reduced via strided Conv2d
    before attention, lowering complexity from O(n²) to O(n²/sr²).
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, sr_ratio: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = float(self.head_dim**-0.5)

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_layer_norm = LayerNorm(hidden_size)

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, height: ir.Value, width: ir.Value
    ):
        query = self.query(op, hidden_states)

        kv_input = hidden_states
        if self.sr_ratio > 1:
            # Reshape to 2D, apply strided conv, flatten back
            batch = op.Shape(hidden_states, start=0, end=1)
            channels = op.Shape(hidden_states, start=2, end=3)
            kv_2d = op.Transpose(hidden_states, perm=[0, 2, 1])
            kv_2d = op.Reshape(kv_2d, op.Concat(batch, channels, height, width, axis=0))
            kv_2d = self.sr(op, kv_2d)
            # Flatten back to [B, S', C]
            batch2 = op.Shape(kv_2d, start=0, end=1)
            ch2 = op.Shape(kv_2d, start=1, end=2)
            kv_2d = op.Reshape(
                kv_2d, op.Concat(batch2, ch2, op.Constant(value_ints=[-1]), axis=0)
            )
            kv_input = op.Transpose(kv_2d, perm=[0, 2, 1])
            kv_input = self.sr_layer_norm(op, kv_input)

        key = self.key(op, kv_input)
        value = self.value(op, kv_input)

        # Reshape to multi-head: [B, S, H] → [B, heads, S, head_dim]
        query = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])
        query = op.Transpose(query, perm=[0, 2, 1, 3])
        key = op.Reshape(key, [0, 0, self.num_heads, self.head_dim])
        key = op.Transpose(key, perm=[0, 2, 1, 3])
        value = op.Reshape(value, [0, 0, self.num_heads, self.head_dim])
        value = op.Transpose(value, perm=[0, 2, 1, 3])

        # Attention: Q @ K^T / sqrt(d) → softmax → @ V
        attn_scores = op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2]))
        attn_scores = op.Mul(attn_scores, op.Constant(value_float=self.scale))
        attn_probs = op.Softmax(attn_scores, axis=-1)
        context = op.MatMul(attn_probs, value)

        # Reshape back: [B, heads, S, head_dim] → [B, S, H]
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        context = op.Reshape(context, [0, 0, self.hidden_size])
        return context


class _DWConv(nn.Module):
    """Depthwise convolution for Mix-FFN position encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, height: ir.Value, width: ir.Value
    ):
        # [B, S, C] → [B, C, H, W]
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=2, end=3)
        x = op.Transpose(hidden_states, perm=[0, 2, 1])
        x = op.Reshape(x, op.Concat(batch, channels, height, width, axis=0))
        x = self.dwconv(op, x)
        # [B, C, H, W] → [B, S, C]
        batch2 = op.Shape(x, start=0, end=1)
        ch2 = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch2, ch2, op.Constant(value_ints=[-1]), axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class _MixFFN(nn.Module):
    """Mix-FFN: Linear → DWConv → GELU → Linear."""

    def __init__(self, hidden_size: int, mlp_ratio: int, hidden_act: str):
        super().__init__()
        mlp_hidden = hidden_size * mlp_ratio
        self.dense1 = Linear(hidden_size, mlp_hidden)
        self.dwconv = _DWConv(mlp_hidden)
        self.dense2 = Linear(mlp_hidden, hidden_size)
        self._act_fn = ACT2FN[hidden_act]

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, height: ir.Value, width: ir.Value
    ):
        hidden_states = self.dense1(op, hidden_states)
        hidden_states = self.dwconv(op, hidden_states, height, width)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.dense2(op, hidden_states)
        return hidden_states


class _SegformerLayer(nn.Module):
    """Segformer encoder layer: pre-norm attention + pre-norm Mix-FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layer_norm_1 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = _EfficientSelfAttention(hidden_size, num_attention_heads, sr_ratio)
        self.output_dense = Linear(hidden_size, hidden_size)
        self.layer_norm_2 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = _MixFFN(hidden_size, mlp_ratio, hidden_act)

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, height: ir.Value, width: ir.Value
    ):
        # Attention block
        normed = self.layer_norm_1(op, hidden_states)
        attn_output = self.attention(op, normed, height, width)
        attn_output = self.output_dense(op, attn_output)
        hidden_states = op.Add(hidden_states, attn_output)

        # MLP block
        normed = self.layer_norm_2(op, hidden_states)
        mlp_output = self.mlp(op, normed, height, width)
        hidden_states = op.Add(hidden_states, mlp_output)
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _SegformerEncoder(nn.Module):
    """Multi-stage hierarchical encoder."""

    def __init__(self, config: SegformerConfig):
        super().__init__()
        hidden_sizes = config.segformer_hidden_sizes or [32, 64, 160, 256]
        num_heads = config.segformer_num_attention_heads or [1, 2, 5, 8]
        depths = config.segformer_depths or [2, 2, 2, 2]
        sr_ratios = config.segformer_sr_ratios or [8, 4, 2, 1]
        mlp_ratios = config.segformer_mlp_ratios or [4, 4, 4, 4]
        patch_sizes = config.segformer_patch_sizes or [7, 3, 3, 3]
        strides = config.segformer_strides or [4, 2, 2, 2]
        num_stages = len(hidden_sizes)
        eps = config.rms_norm_eps
        act = config.hidden_act or "gelu"

        # Patch embeddings per stage
        self.patch_embeddings = nn.ModuleList()
        for i in range(num_stages):
            in_ch = config.num_channels if i == 0 else hidden_sizes[i - 1]
            self.patch_embeddings.append(
                _OverlapPatchEmbeddings(in_ch, hidden_sizes[i], patch_sizes[i], strides[i])
            )

        # Transformer blocks per stage
        self.blocks = nn.ModuleList()
        for i in range(num_stages):
            stage_layers = nn.ModuleList()
            for _ in range(depths[i]):
                stage_layers.append(
                    _SegformerLayer(
                        hidden_sizes[i],
                        num_heads[i],
                        sr_ratios[i],
                        mlp_ratios[i],
                        act,
                        eps,
                    )
                )
            self.blocks.append(stage_layers)

        # Layer norms per stage
        self.layer_norms = nn.ModuleList(
            [LayerNorm(hidden_sizes[i], eps=eps) for i in range(num_stages)]
        )

        self._num_stages = num_stages
        self._hidden_sizes = hidden_sizes

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = pixel_values
        all_hidden_states = []

        for idx in range(self._num_stages):
            # Patch embed
            hidden_states, height, width = self.patch_embeddings[idx](op, hidden_states)

            # Transformer layers
            for layer in self.blocks[idx]:
                hidden_states = layer(op, hidden_states, height, width)

            # Layer norm
            hidden_states = self.layer_norms[idx](op, hidden_states)

            # Reshape to 2D: [B, S, C] → [B, C, H, W]
            batch = op.Shape(hidden_states, start=0, end=1)
            channels = op.Shape(hidden_states, start=2, end=3)
            hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
            hidden_states = op.Reshape(
                hidden_states, op.Concat(batch, channels, height, width, axis=0)
            )
            all_hidden_states.append(hidden_states)

        return all_hidden_states


# ---------------------------------------------------------------------------
# Decode head
# ---------------------------------------------------------------------------


class _LinearCStage(nn.Module):
    """Wrapper to match HF's ``linear_c.N.proj`` weight naming."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = Linear(in_features, out_features)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return self.proj(op, x)


class _SegformerDecodeHead(nn.Module):
    """Lightweight MLP decode head for segmentation."""

    def __init__(self, config: SegformerConfig):
        super().__init__()
        hidden_sizes = config.segformer_hidden_sizes or [32, 64, 160, 256]
        num_stages = len(hidden_sizes)
        dec_hidden = config.decoder_hidden_size
        num_labels = config.num_labels

        # Per-stage linear projection to decoder_hidden_size
        self.linear_c = nn.ModuleList(
            [_LinearCStage(hidden_sizes[i], dec_hidden) for i in range(num_stages)]
        )
        # Fuse concatenated features
        self.linear_fuse = Conv2dNoBias(
            dec_hidden * num_stages,
            dec_hidden,
            kernel_size=1,
        )
        self.batch_norm = BatchNorm2d(dec_hidden)
        # Classifier
        self.classifier = Conv2d(dec_hidden, num_labels, kernel_size=1, padding=0)

    def forward(self, op: builder.OpBuilder, encoder_hidden_states: ir.Value):
        # Target spatial size = first stage output (largest resolution)
        target_h = op.Shape(encoder_hidden_states[0], start=2, end=3)
        target_w = op.Shape(encoder_hidden_states[0], start=3, end=4)

        projected = []
        for hs, mlp in zip(encoder_hidden_states, self.linear_c):
            # [B, C, H, W] → [B, H*W, C]
            batch = op.Shape(hs, start=0, end=1)
            ch = op.Shape(hs, start=1, end=2)
            flat = op.Reshape(hs, op.Concat(batch, ch, op.Constant(value_ints=[-1]), axis=0))
            flat = op.Transpose(flat, perm=[0, 2, 1])
            # Linear projection
            flat = mlp(op, flat)
            # [B, S, dec_hidden] → [B, dec_hidden, H, W]
            dec_ch = op.Shape(flat, start=2, end=3)
            flat = op.Transpose(flat, perm=[0, 2, 1])
            h_i = op.Shape(hs, start=2, end=3)
            w_i = op.Shape(hs, start=3, end=4)
            feat = op.Reshape(flat, op.Concat(batch, dec_ch, h_i, w_i, axis=0))
            # Upsample to target size
            feat = op.Resize(
                feat,
                None,  # roi
                None,  # scales
                op.Concat(op.Shape(feat, start=0, end=2), target_h, target_w, axis=0),
                mode="linear",
            )
            projected.append(feat)

        # Concat in reverse order (coarse to fine) and fuse
        projected.reverse()
        fused = op.Concat(*projected, axis=1)
        fused = self.linear_fuse(op, fused)
        fused = self.batch_norm(op, fused)
        fused = op.Relu(fused)

        logits = self.classifier(op, fused)
        return logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SegformerForSemanticSegmentation(nn.Module):
    """Segformer for semantic segmentation.

    Hierarchical ViT encoder with lightweight MLP decoder.
    Outputs segmentation logits of shape [batch, num_labels, H/4, W/4].
    """

    default_task = "image-classification"
    category = "Segmentation"
    config_class: type = SegformerConfig

    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.encoder = _SegformerEncoder(config)
        self.decode_head = _SegformerDecodeHead(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        encoder_hidden_states = self.encoder(op, pixel_values)
        logits = self.decode_head(op, encoder_hidden_states)
        return logits

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_segformer_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_ENCODER_BLOCK_PATTERN = re.compile(r"^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)$")
_PATCH_EMB_PATTERN = re.compile(r"^segformer\.encoder\.patch_embeddings\.(\d+)\.(.+)$")
_LAYER_NORM_PATTERN = re.compile(r"^segformer\.encoder\.layer_norm\.(\d+)\.(.+)$")

_ATTN_RENAMES = {
    "attention.self.query.": "attention.query.",
    "attention.self.key.": "attention.key.",
    "attention.self.value.": "attention.value.",
    "attention.self.sr.": "attention.sr.",
    "attention.self.sr_layer_norm.": "attention.sr_layer_norm.",
    "attention.output.dense.": "output_dense.",
}

_MLP_RENAMES = {
    "mlp.dense1.": "mlp.dense1.",
    "mlp.dwconv.dwconv.": "mlp.dwconv.dwconv.",
    "mlp.dense2.": "mlp.dense2.",
}


def _rename_segformer_weight(name: str) -> str | None:
    """Rename HF Segformer weight to our naming convention."""
    # Skip classifier head if present at top level
    if name.startswith("classifier."):
        return None

    # Patch embeddings
    m = _PATCH_EMB_PATTERN.match(name)
    if m:
        stage, suffix = m.group(1), m.group(2)
        return f"encoder.patch_embeddings.{stage}.{suffix}"

    # Layer norms
    m = _LAYER_NORM_PATTERN.match(name)
    if m:
        stage, suffix = m.group(1), m.group(2)
        return f"encoder.layer_norms.{stage}.{suffix}"

    # Encoder blocks
    m = _ENCODER_BLOCK_PATTERN.match(name)
    if m:
        stage, layer, suffix = m.group(1), m.group(2), m.group(3)
        base = f"encoder.blocks.{stage}.{layer}"

        for old, new in _ATTN_RENAMES.items():
            if suffix.startswith(old):
                rest = suffix[len(old) :]
                return f"{base}.{new}{rest}"

        for old, new in _MLP_RENAMES.items():
            if suffix.startswith(old):
                rest = suffix[len(old) :]
                return f"{base}.{new}{rest}"

        # layer_norm_1, layer_norm_2 pass through
        if suffix.startswith("layer_norm_"):
            return f"{base}.{suffix}"

        return None

    # Decode head
    if name.startswith("decode_head."):
        suffix = name[len("decode_head.") :]
        return f"decode_head.{suffix}"

    return None
