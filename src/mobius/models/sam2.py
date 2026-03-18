# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SAM2 (Segment Anything Model 2) vision encoder.

SAM2 uses a Hiera backbone (hierarchical ViT) with a Feature Pyramid
Network (FPN) neck. The backbone processes images through multi-scale
stages with increasing channel dimensions.

Simplifications over HuggingFace:
- Global attention (no window partitioning)
- No query stride pooling (spatial resolution constant through backbone)
- No prompt encoder or mask decoder (vision encoder only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Sam2Config
from mobius.components._activations import ACT2FN
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d, Conv2dNoBias

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Hiera backbone components
# ---------------------------------------------------------------------------


class _Sam2HieraBlock(nn.Module):
    """Simplified Hiera block: pre-norm attention + pre-norm FFN.

    Supports dim transitions at stage boundaries (dim_in != dim_out).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        eps: float,
    ):
        super().__init__()
        self.layer_norm1 = LayerNorm(dim_in, eps=eps)
        # Fused QKV projection (matches HF Sam2MultiScaleAttention.qkv)
        self.attn_qkv = Linear(dim_in, dim_out * 3)
        self.attn_proj = Linear(dim_out, dim_out)
        self.layer_norm2 = LayerNorm(dim_out, eps=eps)
        mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp_proj_in = Linear(dim_out, mlp_hidden)
        self.mlp_proj_out = Linear(mlp_hidden, dim_out)
        self._act_fn = ACT2FN[hidden_act]
        self._num_heads = num_heads
        self._head_dim = dim_out // num_heads
        self._has_dim_proj = dim_in != dim_out
        if self._has_dim_proj:
            self.proj = Linear(dim_in, dim_out)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: [B, S, dim_in]
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)

        if self._has_dim_proj:
            residual = self.proj(op, hidden_states)

        # Fused QKV → split
        qkv = self.attn_qkv(op, hidden_states)  # [B, S, 3*dim_out]
        # Reshape to [B, S, 3, num_heads, head_dim]
        new_shape_5d = op.Constant(value_ints=[0, 0, 3, self._num_heads, self._head_dim])
        qkv = op.Reshape(qkv, new_shape_5d)
        # Transpose to [3, B, num_heads, S, head_dim]
        qkv = op.Transpose(qkv, perm=[2, 0, 3, 1, 4])
        q = op.Gather(qkv, op.Constant(value_int=0), axis=0)  # [B, H, S, D]
        k = op.Gather(qkv, op.Constant(value_int=1), axis=0)
        v = op.Gather(qkv, op.Constant(value_int=2), axis=0)

        # Scaled dot-product attention
        scale = self._head_dim**-0.5
        q = op.Mul(q, op.Constant(value_float=scale))
        k_t = op.Transpose(k, perm=[0, 1, 3, 2])
        scores = op.MatMul(q, k_t)
        attn_weights = op.Softmax(scores, axis=-1)
        attn_out = op.MatMul(attn_weights, v)  # [B, H, S, D]

        # Merge heads: [B, S, dim_out]
        attn_out = op.Transpose(attn_out, perm=[0, 2, 1, 3])
        out_shape = op.Shape(residual)
        attn_out = op.Reshape(attn_out, out_shape)

        attn_out = self.attn_proj(op, attn_out)
        hidden_states = op.Add(residual, attn_out)

        # FFN
        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp_proj_in(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.mlp_proj_out(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# SAM2 Vision Model
# ---------------------------------------------------------------------------


class Sam2VisionModel(nn.Module):
    """SAM2 vision encoder: Hiera backbone + FPN neck.

    Outputs the highest-resolution FPN feature map as last_hidden_state.
    """

    default_task = "image-classification"
    category = "vision"
    config_class: type = Sam2Config

    def __init__(self, config: Sam2Config):
        super().__init__()
        embed_dims = config.sam2_embed_dims or [96, 192, 384, 768]
        blocks_per_stage = config.sam2_blocks_per_stage or [1, 2, 7, 2]
        num_heads_per_stage = config.sam2_num_heads_per_stage or [1, 2, 4, 8]
        mlp_ratio = config.sam2_mlp_ratio or 4.0
        fpn_hidden_size = config.sam2_fpn_hidden_size or 256
        hidden_act = config.hidden_act
        eps = config.rms_norm_eps

        initial_dim = embed_dims[0]

        # Patch embedding (Conv2d: 3 → initial_dim, kernel=7, stride=4, pad=3)
        self.patch_embed = Conv2d(
            config.num_channels,
            initial_dim,
            kernel_size=7,
            stride=4,
            padding=3,
        )

        # Learnable position embedding (will be added to flattened patches)
        # Shape matches Hiera: [1, initial_dim, pos_h, pos_w]
        # For graph building, we store it and broadcast-add after flattening
        pos_size = config.image_size // 4  # spatial size after patch embed
        self.pos_embed = nn.Parameter((1, pos_size * pos_size, initial_dim))

        # Build all blocks across stages
        self.blocks = nn.ModuleList()
        stage_ends = []
        total_idx = 0
        for stage_idx, n_blocks in enumerate(blocks_per_stage):
            for block_idx in range(n_blocks):
                # First block of stage > 0 transitions from previous dim
                dim_in = (
                    embed_dims[stage_idx - 1]
                    if stage_idx > 0 and block_idx == 0
                    else embed_dims[stage_idx]
                )
                dim_out = embed_dims[stage_idx]
                num_heads = num_heads_per_stage[stage_idx]
                self.blocks.append(
                    _Sam2HieraBlock(dim_in, dim_out, num_heads, mlp_ratio, hidden_act, eps)
                )
                total_idx += 1
            stage_ends.append(total_idx - 1)

        self._stage_ends = stage_ends
        self._embed_dims = embed_dims
        self._fpn_hidden_size = fpn_hidden_size
        self._pos_size = pos_size

        # FPN neck: one Conv2d per stage to project to fpn_hidden_size
        # backbone_channel_list is reversed embed_dims (high→low resolution)
        self.neck_convs = nn.ModuleList()
        for dim in reversed(embed_dims):
            self.neck_convs.append(
                Conv2dNoBias(dim, fpn_hidden_size, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Patch embedding: [B, 3, H, W] → [B, C, H', W']
        x = self.patch_embed(op, pixel_values)
        # Flatten spatial dims: [B, C, H', W'] → [B, H'*W', C]
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # [B, H', W', C]
        flatten_shape = op.Constant(value_ints=[0, -1, self._embed_dims[0]])
        x = op.Reshape(x, flatten_shape)

        # Add position embedding
        x = op.Add(x, self.pos_embed)

        # Run backbone blocks, collecting intermediate features
        intermediates = []
        for i, block in enumerate(self.blocks):
            x = block(op, x)
            if i in self._stage_ends:
                intermediates.append(x)

        # FPN neck: project each stage to fpn_hidden_size
        # intermediates[0] = stage 0 (dim=embed_dims[0]), ..., intermediates[-1] = last stage
        # Process in reverse order (high to low resolution / large to small dim)
        n_stages = len(intermediates)
        h = self._pos_size
        prev_features = None

        for i in range(n_stages - 1, -1, -1):
            features = intermediates[i]
            dim = self._embed_dims[i]
            # Reshape to [B, C, H, W] for Conv2d
            reshape_4d = op.Constant(value_ints=[0, h, h, dim])
            features = op.Reshape(features, reshape_4d)
            features = op.Transpose(features, perm=[0, 3, 1, 2])  # [B, C, H, W]

            # Apply 1x1 conv to project to fpn_hidden_size
            conv_idx = n_stages - 1 - i
            features = self.neck_convs[conv_idx](op, features)

            if prev_features is not None:
                # Top-down fusion: add upsampled previous features
                features = op.Add(features, prev_features)
            prev_features = features

        # Output: last computed features (highest resolution)
        # Flatten back to [B, S, fpn_hidden_size]
        last_features = op.Transpose(prev_features, perm=[0, 2, 3, 1])  # [B, H, W, C]
        out_shape = op.Constant(value_ints=[0, -1, self._fpn_hidden_size])
        last_hidden_state = op.Reshape(last_features, out_shape)

        return last_hidden_state

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_sam2_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------

# Cumulative block indices for stage_ends calculation
_DEFAULT_BLOCKS_PER_STAGE = [1, 2, 7, 2]


def _rename_sam2_weight(name: str) -> str | None:
    """Rename HF SAM2 weight to our naming convention."""
    # Strip top-level prefixes
    for prefix in ("vision_encoder.backbone.", "backbone."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    else:
        # Handle FPN neck weights
        for prefix in ("vision_encoder.neck.", "neck."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                # convs.{i}.weight → neck_convs.{i}.weight
                if name.startswith("convs."):
                    return f"neck_{name}"
                return None
        # Skip prompt encoder, mask decoder, etc.
        return None

    # patch_embed.projection → patch_embed
    if name.startswith("patch_embed.projection."):
        rest = name[len("patch_embed.projection.") :]
        return f"patch_embed.{rest}"

    # Position embeddings
    if name in ("pos_embed", "pos_embed_window"):
        # TODO(feature): Handle windowed position embedding (pos_embed_window).
        # SAM2 uses window-based attention with separate position embeddings
        # for windowed vs global attention blocks. Currently only the global
        # pos_embed is loaded; pos_embed_window is dropped silently.
        # Prerequisites: Add window partitioning to SAM2 encoder blocks,
        # store pos_embed_window as a parameter, and apply it in windowed
        # attention layers. See HF Sam2VisionAttention for reference.
        # Complexity: M — requires window partition/unpartition ops + config
        # for which layers use windowed vs global attention.
        if name == "pos_embed":
            return "pos_embed"
        return None

    # blocks.{i}.* → blocks.{i}.*
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        block_idx = parts[1]
        remainder = parts[2]

        # Attention: attn.qkv → attn_qkv, attn.proj → attn_proj
        if remainder.startswith("attn.qkv."):
            return f"blocks.{block_idx}.attn_qkv.{remainder[len('attn.qkv.') :]}"
        if remainder.startswith("attn.proj."):
            return f"blocks.{block_idx}.attn_proj.{remainder[len('attn.proj.') :]}"

        # FFN: mlp.proj_in → mlp_proj_in, mlp.proj_out → mlp_proj_out
        if remainder.startswith("mlp.proj_in."):
            return f"blocks.{block_idx}.mlp_proj_in.{remainder[len('mlp.proj_in.') :]}"
        if remainder.startswith("mlp.proj_out."):
            return f"blocks.{block_idx}.mlp_proj_out.{remainder[len('mlp.proj_out.') :]}"
        # mlp.layers not used (only 2-layer MLP has proj_in + proj_out)

        # LayerNorm, proj pass through
        if remainder.startswith(("layer_norm1.", "layer_norm2.", "proj.")):
            return f"blocks.{block_idx}.{remainder}"

        return None

    return None
