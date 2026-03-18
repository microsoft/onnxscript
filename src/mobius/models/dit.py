# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DiT (Diffusion Transformer) 2D model.

Pure transformer architecture for diffusion denoising. Used by PixArt-alpha,
DiT-XL, and similar models.

Architecture:
1. Patch embedding (2D → sequence of patches)
2. AdaLN-Zero transformer blocks with cross-attention
3. Unpatchify (sequence → 2D)

HF diffusers class: PixArtTransformer2DModel / DiTTransformer2DModel
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


@dataclasses.dataclass
class DiTConfig:
    """Configuration for DiT/PixArt transformer denoisers."""

    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 1152
    num_layers: int = 28
    num_attention_heads: int = 16
    cross_attention_dim: int = 768
    caption_channels: int = 768
    sample_size: int = 64
    norm_eps: float = 1e-6
    act_fn: str = "gelu_tanh"

    @classmethod
    def from_diffusers(cls, config: dict) -> DiTConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 4),
            patch_size=config.get("patch_size", 2),
            hidden_size=config.get("hidden_size", 1152),
            num_layers=config.get("num_layers", 28),
            num_attention_heads=config.get("num_attention_heads", 16),
            cross_attention_dim=config.get("cross_attention_dim", 768),
            caption_channels=config.get("caption_channels", 768),
            sample_size=config.get("sample_size", 64),
            norm_eps=config.get("norm_eps", 1e-6),
        )


# ---------------------------------------------------------------------------
# Components (extracted to components/_diffusion.py; remaining model-specific)
# ---------------------------------------------------------------------------


class _DiTCrossAttention(nn.Module):
    """Multi-head cross-attention: Q from latent, KV from encoder output."""

    def __init__(self, hidden_size: int, cross_attention_dim: int, num_heads: int):
        super().__init__()
        self.to_q = _Linear(hidden_size, hidden_size)
        self.to_k = _Linear(cross_attention_dim, hidden_size)
        self.to_v = _Linear(cross_attention_dim, hidden_size)
        self.to_out = nn.Sequential(_Linear(hidden_size, hidden_size))
        self.norm = _LayerNorm(hidden_size)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, encoder_hidden_states: ir.Value
    ):
        normed = self.norm(op, hidden_states)
        q = self.to_q(op, normed)
        k = self.to_k(op, encoder_hidden_states)
        v = self.to_v(op, encoder_hidden_states)
        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return op.Add(self.to_out(op, attn_out), hidden_states)


class _DiTBlock(nn.Module):
    """DiT transformer block with AdaLN-Zero modulation."""

    def __init__(
        self, hidden_size: int, num_heads: int, cross_attention_dim: int, eps: float = 1e-6
    ):
        super().__init__()
        self.norm1 = _AdaLayerNormZero(hidden_size, eps=eps)
        self.attn1 = _DiTSelfAttention(hidden_size, num_heads)
        self.attn2 = _DiTCrossAttention(hidden_size, cross_attention_dim, num_heads)
        self.ff = _DiTFFN(hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        temb: ir.Value,
    ):
        # AdaLN-Zero modulation
        normed, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            op,
            hidden_states,
            temb,
        )

        # Apply scale and shift to normed hidden_states for self-attention
        one = op.Constant(value_float=1.0)
        attn_input = op.Mul(normed, op.Add(one, op.Unsqueeze(scale_msa, [1])))
        attn_input = op.Add(attn_input, op.Unsqueeze(shift_msa, [1]))

        # Self-attention with gate
        attn_out = self.attn1(op, attn_input)
        attn_out = op.Mul(attn_out, op.Unsqueeze(gate_msa, [1]))
        hidden_states = op.Add(hidden_states, attn_out)

        # Cross-attention
        hidden_states = self.attn2(op, hidden_states, encoder_hidden_states)

        # FFN with AdaLN modulation
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

        return hidden_states


# ---------------------------------------------------------------------------
# Full DiT model
# ---------------------------------------------------------------------------


class DiTTransformer2DModel(nn.Module):
    """DiT transformer denoiser for latent diffusion.

    Patch-based transformer with AdaLN-Zero conditioning on timestep and
    cross-attention conditioning on text encoder output.
    """

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        patch_size = config.patch_size

        # Patch embedding: Conv2d with stride=patch_size
        self.pos_embed = _PatchEmbed(
            config.in_channels,
            hidden_size,
            patch_size,
        )

        # Time embedding
        self.time_proj_dim = hidden_size
        self.adaln_single = _TimestepEmbedding(hidden_size, hidden_size)

        # Caption projection (text encoder → hidden size)
        self.caption_projection = _Linear(config.caption_channels, hidden_size)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.transformer_blocks.append(
                _DiTBlock(
                    hidden_size,
                    config.num_attention_heads,
                    hidden_size,
                    eps=config.norm_eps,
                )
            )

        # Output: AdaLN → Linear → unpatchify
        self.norm_out = _AdaLayerNormOutput(hidden_size, eps=config.norm_eps)
        self.proj_out = _Linear(hidden_size, patch_size * patch_size * config.out_channels)

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            sample: Noisy latent [batch, in_channels, height, width]
            timestep: Diffusion timestep [batch]
            encoder_hidden_states: Text encoder output [batch, seq_len, caption_channels]

        Returns:
            noise_pred: [batch, out_channels, height, width]
        """
        # Patch embedding
        hidden_states = self.pos_embed(op, sample)

        # Time embedding
        t_emb = self._get_timestep_embedding(op, timestep)
        temb = self.adaln_single(op, t_emb)

        # Project text encoder output
        encoder_hidden_states = self.caption_projection(op, encoder_hidden_states)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(op, hidden_states, encoder_hidden_states, temb)

        # Output normalization + projection
        hidden_states = self.norm_out(op, hidden_states, temb)
        hidden_states = self.proj_out(op, hidden_states)

        # Unpatchify [B, num_patches, patch_size*patch_size*C] → [B, C, H, W]
        hidden_states = self._unpatchify(op, hidden_states, sample)

        return hidden_states

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        """Sinusoidal timestep embedding."""
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        return op.Concat(op.Cos(args), op.Sin(args), axis=-1)

    def _unpatchify(self, op: builder.OpBuilder, hidden_states, original_input):
        """Reshape patches back to spatial dimensions."""
        p = self.config.patch_size
        c = self.config.out_channels
        batch = op.Shape(original_input, start=0, end=1)
        h = op.Shape(original_input, start=2, end=3)
        w = op.Shape(original_input, start=3, end=4)
        h_patches = op.Div(h, op.Constant(value_ints=[p]))
        w_patches = op.Div(w, op.Constant(value_ints=[p]))

        # [B, num_patches, p*p*c] → [B, h_p, w_p, p, p, c]
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                h_patches,
                w_patches,
                op.Constant(value_ints=[p, p, c]),
                axis=0,
            ),
        )
        # [B, h_p, w_p, p, p, c] → [B, c, h_p, p, w_p, p]
        hidden_states = op.Transpose(hidden_states, perm=[0, 5, 1, 3, 2, 4])
        # [B, c, h_p*p, w_p*p] = [B, c, H, W]
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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
