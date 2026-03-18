# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SD3 and Flux transformer denoisers.

SD3 (MMDiT): Multimodal DiT with joint attention between text and image tokens.
Flux: Double-stream + single-stream transformer with flow matching.

Both use AdaLN-Zero modulation from timestep conditioning.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components import LayerNorm as _LayerNorm
from mobius.components import Linear as _Linear
from mobius.components._diffusion import (
    AdaLayerNormOutput as _AdaLayerNormOutput,
)
from mobius.components._diffusion import (
    AdaLayerNormZero as _AdaLayerNormZero,
)
from mobius.components._diffusion import (
    DiffusionFFN as _DiTFFN,
)
from mobius.components._diffusion import (
    DiffusionSelfAttention as _DiTSelfAttention,
)
from mobius.components._diffusion import (
    PatchEmbed as _PatchEmbed,
)
from mobius.components._diffusion import (
    TimestepEmbedding as _TimestepEmbedding,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# SD3 (MMDiT)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SD3Config:
    """Configuration for SD3 Transformer (MMDiT)."""

    in_channels: int = 16
    out_channels: int = 16
    patch_size: int = 2
    hidden_size: int = 1536
    num_layers: int = 24
    num_attention_heads: int = 24
    pooled_projection_dim: int = 2048
    caption_projection_dim: int = 1536
    joint_attention_dim: int = 4096
    sample_size: int = 128
    norm_eps: float = 1e-6
    # Used for denoising task
    cross_attention_dim: int = 4096

    @classmethod
    def from_diffusers(cls, config: dict) -> SD3Config:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 16),
            out_channels=config.get("out_channels", 16),
            patch_size=config.get("patch_size", 2),
            hidden_size=config.get("hidden_size", 1536),
            num_layers=config.get("num_layers", 24),
            num_attention_heads=config.get("num_attention_heads", 24),
            pooled_projection_dim=config.get("pooled_projection_dim", 2048),
            caption_projection_dim=config.get("caption_projection_dim", 1536),
            joint_attention_dim=config.get("joint_attention_dim", 4096),
            sample_size=config.get("sample_size", 128),
        )


class _JointAttentionBlock(nn.Module):
    """MMDiT joint attention: concatenate text + image tokens, attend jointly."""

    def __init__(self, hidden_size: int, num_heads: int, context_dim: int, eps: float = 1e-6):
        super().__init__()
        # Image stream
        self.norm1 = _AdaLayerNormZero(hidden_size, eps=eps)
        self.attn = _DiTSelfAttention(hidden_size, num_heads)
        self.ff = _DiTFFN(hidden_size)

        # Context (text) stream: project context to hidden_size for joint attention
        self.norm1_context = _LayerNorm(context_dim, eps=eps)
        self.context_proj_in = (
            _Linear(context_dim, hidden_size) if context_dim != hidden_size else None
        )
        self.context_proj_out = (
            _Linear(hidden_size, context_dim) if context_dim != hidden_size else None
        )
        self.norm_context_ff = _LayerNorm(context_dim, eps=eps)
        self.ff_context = _DiTFFN(context_dim)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        temb: ir.Value,
    ):
        # AdaLN-Zero modulation for image stream
        normed, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            op,
            hidden_states,
            temb,
        )
        one = op.Constant(value_float=1.0)
        img_input = op.Mul(normed, op.Add(one, op.Unsqueeze(scale_msa, [1])))
        img_input = op.Add(img_input, op.Unsqueeze(shift_msa, [1]))

        # Normalize context
        ctx_normed = self.norm1_context(op, encoder_hidden_states)
        if self.context_proj_in is not None:
            ctx_normed = self.context_proj_in(op, ctx_normed)

        # Joint attention: concatenate image + context tokens
        joint_input = op.Concat(img_input, ctx_normed, axis=1)
        joint_output = self.attn(op, joint_input)

        # Split back
        img_seq_len = op.Shape(img_input, start=1, end=2)
        ctx_seq_len = op.Shape(ctx_normed, start=1, end=2)
        img_attn, ctx_attn = op.Split(
            joint_output, op.Concat(img_seq_len, ctx_seq_len, axis=0), axis=1, _outputs=2
        )

        # Apply gate to image
        img_attn = op.Mul(img_attn, op.Unsqueeze(gate_msa, [1]))
        hidden_states = op.Add(hidden_states, img_attn)

        # FFN for image
        normed_ff = op.LayerNormalization(
            hidden_states,
            self.norm1.norm.weight,
            self.norm1.norm.bias,
            axis=-1,
            epsilon=self.norm1.norm.eps,
        )
        ff_input = op.Mul(normed_ff, op.Add(one, op.Unsqueeze(scale_mlp, [1])))
        ff_input = op.Add(ff_input, op.Unsqueeze(shift_mlp, [1]))
        ff_out = self.ff(op, ff_input)
        ff_out = op.Mul(ff_out, op.Unsqueeze(gate_mlp, [1]))
        hidden_states = op.Add(hidden_states, ff_out)

        # Context stream: add attention + FFN
        if self.context_proj_out is not None:
            ctx_attn = self.context_proj_out(op, ctx_attn)
        encoder_hidden_states = op.Add(encoder_hidden_states, ctx_attn)

        ctx_ff = self.norm_context_ff(op, encoder_hidden_states)
        ctx_ff = self.ff_context(op, ctx_ff)
        encoder_hidden_states = op.Add(encoder_hidden_states, ctx_ff)

        return hidden_states, encoder_hidden_states


class SD3Transformer2DModel(nn.Module):
    """SD3 (MMDiT) transformer denoiser with joint text-image attention."""

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: SD3Config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.pos_embed = _PatchEmbed(config.in_channels, hidden_size, config.patch_size)
        self.time_proj_dim = hidden_size
        self.time_text_embed = _TimestepEmbedding(hidden_size, hidden_size)
        self.context_embedder = _Linear(config.joint_attention_dim, config.joint_attention_dim)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.transformer_blocks.append(
                _JointAttentionBlock(
                    hidden_size,
                    config.num_attention_heads,
                    config.joint_attention_dim,
                    eps=config.norm_eps,
                )
            )

        self.norm_out = _AdaLayerNormOutput(hidden_size, eps=config.norm_eps)
        self.proj_out = _Linear(
            hidden_size, config.patch_size * config.patch_size * config.out_channels
        )

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        hidden_states = self.pos_embed(op, sample)

        t_emb = self._get_timestep_embedding(op, timestep)
        temb = self.time_text_embed(op, t_emb)
        encoder_hidden_states = self.context_embedder(op, encoder_hidden_states)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                op,
                hidden_states,
                encoder_hidden_states,
                temb,
            )

        hidden_states = self.norm_out(op, hidden_states, temb)
        hidden_states = self.proj_out(op, hidden_states)

        # Unpatchify
        p = self.config.patch_size
        c = self.config.out_channels
        batch = op.Shape(sample, start=0, end=1)
        h = op.Shape(sample, start=2, end=3)
        w = op.Shape(sample, start=3, end=4)
        h_patches = op.Div(h, op.Constant(value_ints=[p]))
        w_patches = op.Div(w, op.Constant(value_ints=[p]))

        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, h_patches, w_patches, op.Constant(value_ints=[p, p, c]), axis=0),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 5, 1, 3, 2, 4])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Constant(value_ints=[c]),
                op.Mul(h_patches, op.Constant(value_ints=[p])),
                op.Mul(w_patches, op.Constant(value_ints=[p])),
                axis=0,
            ),
        )
        return hidden_states

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        return op.Concat(op.Cos(args), op.Sin(args), axis=-1)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict


# ---------------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FluxConfig:
    """Configuration for Flux transformer denoiser."""

    in_channels: int = 64
    out_channels: int = 64
    patch_size: int = 1
    hidden_size: int = 3072
    num_layers: int = 19  # Double-stream blocks
    num_single_layers: int = 38  # Single-stream blocks
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    guidance_embeds: bool = False
    sample_size: int = 128
    norm_eps: float = 1e-6
    # For denoising task
    cross_attention_dim: int = 4096

    @classmethod
    def from_diffusers(cls, config: dict) -> FluxConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 64),
            out_channels=config.get("out_channels", 64),
            patch_size=config.get("patch_size", 1),
            hidden_size=config.get("hidden_size", 3072),
            num_layers=config.get("num_layers", 19),
            num_single_layers=config.get("num_single_layers", 38),
            num_attention_heads=config.get("num_attention_heads", 24),
            joint_attention_dim=config.get("joint_attention_dim", 4096),
            guidance_embeds=config.get("guidance_embeds", False),
            sample_size=config.get("sample_size", 128),
        )


class _FluxSingleBlock(nn.Module):
    """Flux single-stream transformer block: unified self-attention + FFN."""

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm = _AdaLayerNormZero(hidden_size, eps=eps)
        self.attn = _DiTSelfAttention(hidden_size, num_heads)
        self.ff = _DiTFFN(hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        normed, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm(
            op,
            hidden_states,
            temb,
        )
        one = op.Constant(value_float=1.0)
        attn_input = op.Mul(normed, op.Add(one, op.Unsqueeze(scale_msa, [1])))
        attn_input = op.Add(attn_input, op.Unsqueeze(shift_msa, [1]))

        attn_out = self.attn(op, attn_input)
        attn_out = op.Mul(attn_out, op.Unsqueeze(gate_msa, [1]))
        hidden_states = op.Add(hidden_states, attn_out)

        normed_ff = op.LayerNormalization(
            hidden_states,
            self.norm.norm.weight,
            self.norm.norm.bias,
            axis=-1,
            epsilon=self.norm.norm.eps,
        )
        ff_input = op.Mul(normed_ff, op.Add(one, op.Unsqueeze(scale_mlp, [1])))
        ff_input = op.Add(ff_input, op.Unsqueeze(shift_mlp, [1]))
        ff_out = self.ff(op, ff_input)
        ff_out = op.Mul(ff_out, op.Unsqueeze(gate_mlp, [1]))
        hidden_states = op.Add(hidden_states, ff_out)

        return hidden_states


class FluxTransformer2DModel(nn.Module):
    """Flux transformer denoiser: double-stream + single-stream blocks."""

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: FluxConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.pos_embed = _PatchEmbed(config.in_channels, hidden_size, config.patch_size)
        self.time_proj_dim = hidden_size
        self.time_text_embed = _TimestepEmbedding(hidden_size, hidden_size)
        self.context_embedder = _Linear(config.joint_attention_dim, hidden_size)

        # Double-stream blocks (joint attention)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.transformer_blocks.append(
                _JointAttentionBlock(
                    hidden_size, config.num_attention_heads, hidden_size, eps=config.norm_eps
                )
            )

        # Single-stream blocks
        self.single_transformer_blocks = nn.ModuleList()
        for _ in range(config.num_single_layers):
            self.single_transformer_blocks.append(
                _FluxSingleBlock(hidden_size, config.num_attention_heads, eps=config.norm_eps)
            )

        self.norm_out = _AdaLayerNormOutput(hidden_size, eps=config.norm_eps)
        self.proj_out = _Linear(
            hidden_size, config.patch_size * config.patch_size * config.out_channels
        )

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        hidden_states = self.pos_embed(op, sample)

        t_emb = self._get_timestep_embedding(op, timestep)
        temb = self.time_text_embed(op, t_emb)
        encoder_hidden_states = self.context_embedder(op, encoder_hidden_states)

        # Double-stream: joint attention
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                op,
                hidden_states,
                encoder_hidden_states,
                temb,
            )

        # Concatenate image + context for single-stream
        hidden_states = op.Concat(hidden_states, encoder_hidden_states, axis=1)

        # Single-stream: unified self-attention
        for block in self.single_transformer_blocks:
            hidden_states = block(op, hidden_states, temb)

        # Take only image tokens (drop context suffix)
        img_seq_len = op.Shape(sample, start=2, end=3)
        img_w = op.Shape(sample, start=3, end=4)
        p = self.config.patch_size
        num_patches = op.Div(op.Mul(img_seq_len, img_w), op.Constant(value_ints=[p * p]))
        batch = op.Shape(hidden_states, start=0, end=1)
        ch = op.Shape(hidden_states, start=2, end=3)  # noqa: F841
        hidden_states = op.Slice(
            hidden_states,
            op.Constant(value_ints=[0]),
            num_patches,
            op.Constant(value_ints=[1]),
        )

        hidden_states = self.norm_out(op, hidden_states, temb)
        hidden_states = self.proj_out(op, hidden_states)

        # Unpatchify
        c = self.config.out_channels
        h = op.Shape(sample, start=2, end=3)
        w = op.Shape(sample, start=3, end=4)
        h_patches = op.Div(h, op.Constant(value_ints=[p]))
        w_patches = op.Div(w, op.Constant(value_ints=[p]))

        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, h_patches, w_patches, op.Constant(value_ints=[p, p, c]), axis=0),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 5, 1, 3, 2, 4])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Constant(value_ints=[c]),
                op.Mul(h_patches, op.Constant(value_ints=[p])),
                op.Mul(w_patches, op.Constant(value_ints=[p])),
                axis=0,
            ),
        )
        return hidden_states

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        return op.Concat(op.Cos(args), op.Sin(args), axis=-1)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
