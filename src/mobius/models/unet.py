# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""UNet2DConditionModel for Stable Diffusion denoisers.

Architecture:
1. Time embedding: sinusoidal → MLP
2. Conv_in: projects noisy latent
3. Down blocks: ResNet + cross-attention + downsample
4. Mid block: ResNet + cross-attention + ResNet
5. Up blocks: ResNet + cross-attention + upsample
6. Conv_out: projects to noise prediction

Supports: SD 1.x, SD 2.x, SDXL UNet
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._diffusers_configs import UNet2DConfig
from mobius.components import Conv2d as _Conv2d
from mobius.components import GroupNorm as _GroupNorm
from mobius.components import Linear as _Linear
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Time Embedding
# ---------------------------------------------------------------------------


class _TimestepEmbedding(nn.Module):
    """Projects timestep embedding to model hidden dim: Linear → SiLU → Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = _Linear(in_channels, time_embed_dim)
        self.linear_2 = _Linear(time_embed_dim, time_embed_dim)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, sample: ir.Value):
        sample = self.linear_1(op, sample)
        sample = self._silu(op, sample)
        sample = self.linear_2(op, sample)
        return sample


# ---------------------------------------------------------------------------
# ResNet with time embedding
# ---------------------------------------------------------------------------


class _ResNetBlock2DWithTime(nn.Module):
    """ResNet block with time embedding injection.

    GroupNorm → SiLU → Conv → time_proj → GroupNorm → SiLU → Conv + skip.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = _GroupNorm(norm_num_groups, in_channels)
        self.conv1 = _Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = _Linear(time_embed_dim, out_channels)
        self.norm2 = _GroupNorm(norm_num_groups, out_channels)
        self.conv2 = _Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self._silu = _SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = _Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        residual = hidden_states

        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv1(op, hidden_states)

        # Add time embedding: [B, C] → [B, C, 1, 1]
        temb_proj = self._silu(op, temb)
        temb_proj = self.time_emb_proj(op, temb_proj)
        temb_proj = op.Unsqueeze(temb_proj, [-1, -2])
        hidden_states = op.Add(hidden_states, temb_proj)

        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv2(op, hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(op, residual)

        return op.Add(hidden_states, residual)


# ---------------------------------------------------------------------------
# Cross-Attention for conditioning
# ---------------------------------------------------------------------------


class _CrossAttentionBlock(nn.Module):
    """Cross-attention block: self-attention + cross-attention + FFN.

    Processes latent features conditioned on text encoder hidden states.
    """

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int,
        num_heads: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.norm = _GroupNorm(norm_num_groups, channels)
        # Self-attention
        self.attn1 = _BasicAttention(channels, channels, num_heads)
        # Cross-attention (K, V from encoder_hidden_states)
        self.attn2 = _BasicAttention(channels, cross_attention_dim, num_heads)
        # FFN
        self.ff = _FeedForward(channels, channels * 4)
        self.norm1 = _LayerNorm1D(channels)
        self.norm2 = _LayerNorm1D(channels)
        self.norm3 = _LayerNorm1D(channels)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        residual = hidden_states
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=1, end=2)
        height = op.Shape(hidden_states, start=2, end=3)
        width = op.Shape(hidden_states, start=3, end=4)

        hidden_states = self.norm(op, hidden_states)

        # Reshape [B, C, H, W] → [B, H*W, C]
        spatial = op.Mul(height, width)
        hidden_states = op.Reshape(hidden_states, op.Concat(batch, channels, spatial, axis=0))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Self-attention
        norm_hs = self.norm1(op, hidden_states)
        hidden_states = op.Add(self.attn1(op, norm_hs, norm_hs), hidden_states)

        # Cross-attention
        norm_hs = self.norm2(op, hidden_states)
        context = encoder_hidden_states if encoder_hidden_states is not None else norm_hs
        hidden_states = op.Add(self.attn2(op, norm_hs, context), hidden_states)

        # FFN
        norm_hs = self.norm3(op, hidden_states)
        hidden_states = op.Add(self.ff(op, norm_hs), hidden_states)

        # Reshape back [B, H*W, C] → [B, C, H, W]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.Reshape(
            hidden_states, op.Concat(batch, channels, height, width, axis=0)
        )

        return op.Add(hidden_states, residual)


class _BasicAttention(nn.Module):
    """Simple multi-head attention: Q from input, K/V from context."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.to_q = _Linear(query_dim, query_dim)
        self.to_k = _Linear(context_dim, query_dim)
        self.to_v = _Linear(context_dim, query_dim)
        self.to_out = nn.Sequential(_Linear(query_dim, query_dim))
        self._num_heads = num_heads
        self._head_dim = query_dim // num_heads

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, context: ir.Value):
        q = self.to_q(op, hidden_states)
        k = self.to_k(op, context)
        v = self.to_v(op, context)

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return self.to_out(op, attn_out)


class _LayerNorm1D(nn.Module):
    """Layer normalization for sequence data [B, T, C]."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self.bias = nn.Parameter((dim,))
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.LayerNormalization(x, self.weight, self.bias, axis=-1, epsilon=self._eps)


class _FeedForward(nn.Module):
    """GEGLU feed-forward network for transformer blocks."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        # GEGLU: projects to 2*inner_dim, splits into value and gate
        self.proj_in = _Linear(dim, inner_dim * 2)
        self.proj_out = _Linear(inner_dim, dim)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        projected = self.proj_in(op, x)
        # Split into value and gate
        x1, gate = op.Split(projected, num_outputs=2, axis=-1, _outputs=2)
        # GELU gate
        gate = op.Gelu(gate)
        hidden_states = op.Mul(x1, gate)
        return self.proj_out(op, hidden_states)


# ---------------------------------------------------------------------------
# Down / Up blocks
# ---------------------------------------------------------------------------


class _DownBlock2D(nn.Module):
    """UNet down block: N (ResNet + optional attention) + downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_head_dim: int = 8,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if cross_attention_dim else None

        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                _ResNetBlock2DWithTime(res_in, out_channels, time_embed_dim, norm_num_groups)
            )
            if cross_attention_dim:
                num_heads = max(1, out_channels // attention_head_dim)
                self.attentions.append(
                    _CrossAttentionBlock(
                        out_channels, cross_attention_dim, num_heads, norm_num_groups
                    )
                )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList()
            self.downsamplers.append(_Downsample2D(out_channels))

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        output_states = []
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(op, hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](op, hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(op, hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class _UpBlock2D(nn.Module):
    """UNet up block: N (ResNet + optional attention) + upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channels: int,
        time_embed_dim: int,
        num_layers: int = 3,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_head_dim: int = 8,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if cross_attention_dim else None

        for i in range(num_layers):
            # First layer takes concatenated skip connection
            res_skip_channels = prev_output_channels if i == 0 else out_channels
            res_in = (in_channels if i == 0 else out_channels) + res_skip_channels
            self.resnets.append(
                _ResNetBlock2DWithTime(res_in, out_channels, time_embed_dim, norm_num_groups)
            )
            if cross_attention_dim:
                num_heads = max(1, out_channels // attention_head_dim)
                self.attentions.append(
                    _CrossAttentionBlock(
                        out_channels, cross_attention_dim, num_heads, norm_num_groups
                    )
                )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList()
            self.upsamplers.append(_Upsample2D(out_channels))

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        skip_connections: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        for i, resnet in enumerate(self.resnets):
            skip = skip_connections.pop()
            hidden_states = op.Concat(hidden_states, skip, axis=1)
            hidden_states = resnet(op, hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](op, hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(op, hidden_states)

        return hidden_states


class _Downsample2D(nn.Module):
    """Strided convolution downsampler."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return self.conv(op, x)


class _Upsample2D(nn.Module):
    """Nearest-neighbor upsampling + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Upsample 2x using nearest neighbor
        hidden_states = op.Resize(
            hidden_states,
            None,
            None,
            op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
            mode="nearest",
        )
        return self.conv(op, hidden_states)


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------


class _UNetMidBlock2DCrossAttn(nn.Module):
    """UNet mid block: ResNet + cross-attention + ResNet."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        cross_attention_dim: int,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        num_heads = max(1, channels // attention_head_dim)
        self.resnets = nn.ModuleList()
        self.resnets.append(
            _ResNetBlock2DWithTime(channels, channels, time_embed_dim, norm_num_groups)
        )
        self.resnets.append(
            _ResNetBlock2DWithTime(channels, channels, time_embed_dim, norm_num_groups)
        )
        self.attentions = nn.ModuleList()
        self.attentions.append(
            _CrossAttentionBlock(channels, cross_attention_dim, num_heads, norm_num_groups)
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        hidden_states = self.resnets[0](op, hidden_states, temb)
        hidden_states = self.attentions[0](op, hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](op, hidden_states, temb)
        return hidden_states


# ---------------------------------------------------------------------------
# Full UNet model
# ---------------------------------------------------------------------------


class UNet2DConditionModel(nn.Module):
    """UNet2D conditional denoiser for Stable Diffusion.

    Takes noisy latent + timestep + text encoder hidden states, outputs noise prediction.
    """

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: UNet2DConfig):
        super().__init__()
        self.config = config

        block_out_channels = config.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # Time embedding
        self.time_proj_dim = block_out_channels[0]
        self.time_embedding = _TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Input convolution
        self.conv_in = _Conv2d(
            config.in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = ch
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(
                _DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    time_embed_dim=time_embed_dim,
                    num_layers=config.layers_per_block,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=config.cross_attention_dim,
                    attention_head_dim=config.attention_head_dim,
                    add_downsample=not is_final,
                )
            )

        # Mid block
        self.mid_block = _UNetMidBlock2DCrossAttn(
            channels=block_out_channels[-1],
            time_embed_dim=time_embed_dim,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
            norm_num_groups=config.norm_num_groups,
        )

        # Up blocks (reversed)
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = ch
            prev_output_channel = reversed_channels[min(i + 1, len(reversed_channels) - 1)]
            is_final = i == len(reversed_channels) - 1
            self.up_blocks.append(
                _UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channels=prev_output_channel,
                    time_embed_dim=time_embed_dim,
                    num_layers=config.layers_per_block + 1,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=config.cross_attention_dim,
                    attention_head_dim=config.attention_head_dim,
                    add_upsample=not is_final,
                )
            )

        # Output
        self.conv_norm_out = _GroupNorm(config.norm_num_groups, block_out_channels[0])
        self.conv_out = _Conv2d(
            block_out_channels[0], config.out_channels, kernel_size=3, padding=1
        )
        self._silu = _SiLU()

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        """Forward pass for denoising.

        Args:
            op: ONNX op builder.
            sample: Noisy latent [batch, in_channels, height, width]
            timestep: Diffusion timestep [batch]
            encoder_hidden_states: Text encoder output [batch, seq_len, cross_dim]

        Returns:
            noise_pred: Predicted noise [batch, out_channels, height, width]
        """
        # Time embedding: sinusoidal position encoding + MLP
        # Using half_dim = dim // 2 sinusoidal encoding
        t_emb = self._get_timestep_embedding(op, timestep)
        emb = self.time_embedding(op, t_emb)

        # Input conv
        sample = self.conv_in(op, sample)

        # Down
        down_block_res_samples = [sample]
        for down_block in self.down_blocks:
            sample, res_samples = down_block(op, sample, emb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)

        # Mid
        sample = self.mid_block(op, sample, emb, encoder_hidden_states)

        # Up
        for up_block in self.up_blocks:
            sample = up_block(op, sample, emb, down_block_res_samples, encoder_hidden_states)

        # Output
        sample = self.conv_norm_out(op, sample)
        sample = self._silu(op, sample)
        sample = self.conv_out(op, sample)

        return sample

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        """Sinusoidal timestep embedding."""
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        # Create frequency array as constant
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())

        # timestep: [batch] → [batch, 1]
        t = op.Cast(timestep, to=1)  # FLOAT
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        embedding = op.Concat(op.Cos(args), op.Sin(args), axis=-1)
        return embedding

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
