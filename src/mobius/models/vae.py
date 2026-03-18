# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""AutoencoderKL (VAE) model for diffusers.

Provides encoder and decoder for latent diffusion models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._diffusers_configs import VAEConfig
from mobius.components import Conv2d as _Conv2d
from mobius.components import GroupNorm as _GroupNorm
from mobius.components import Linear as _Linear
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# ResNet block
# ---------------------------------------------------------------------------


class _ResNetBlock2D(nn.Module):
    """ResNet block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = _GroupNorm(norm_num_groups, in_channels)
        self.conv1 = _Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _GroupNorm(norm_num_groups, out_channels)
        self.conv2 = _Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self._silu = _SiLU()

        # Channel change requires a shortcut conv
        if in_channels != out_channels:
            self.conv_shortcut = _Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states

        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv1(op, hidden_states)

        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv2(op, hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(op, residual)

        return op.Add(hidden_states, residual)


# ---------------------------------------------------------------------------
# Attention block for mid-block
# ---------------------------------------------------------------------------


class _AttentionBlock(nn.Module):
    """Self-attention in latent space with GroupNorm."""

    def __init__(self, channels: int, num_head_channels: int = 1, norm_num_groups: int = 32):
        super().__init__()
        self.group_norm = _GroupNorm(norm_num_groups, channels)
        self.to_q = _Linear(channels, channels)
        self.to_k = _Linear(channels, channels)
        self.to_v = _Linear(channels, channels)
        self.to_out = nn.Sequential(_Linear(channels, channels))
        self._channels = channels
        self._num_heads = max(1, channels // num_head_channels) if num_head_channels > 0 else 1

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=1, end=2)
        height = op.Shape(hidden_states, start=2, end=3)
        width = op.Shape(hidden_states, start=3, end=4)

        hidden_states = self.group_norm(op, hidden_states)

        # Reshape [B, C, H, W] → [B, C, H*W] → [B, H*W, C]
        spatial = op.Mul(height, width)
        hidden_states = op.Reshape(hidden_states, op.Concat(batch, channels, spatial, axis=0))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        query = self.to_q(op, hidden_states)
        key = self.to_k(op, hidden_states)
        value = self.to_v(op, hidden_states)

        # Simple scaled dot-product attention
        scale = op.Constant(value_float=float(self._channels**-0.5))
        query = op.Mul(query, scale)
        attn_weights = op.MatMul(query, op.Transpose(key, perm=[0, 2, 1]))
        attn_weights = op.Softmax(attn_weights, axis=-1)
        hidden_states = op.MatMul(attn_weights, value)

        hidden_states = self.to_out(op, hidden_states)

        # Reshape back [B, H*W, C] → [B, C, H, W]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.Reshape(
            hidden_states, op.Concat(batch, channels, height, width, axis=0)
        )

        return op.Add(hidden_states, residual)


# ---------------------------------------------------------------------------
# Encoder blocks
# ---------------------------------------------------------------------------


class _DownEncoderBlock2D(nn.Module):
    """Down-sampling encoder block: N ResNets + optional downsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(_ResNetBlock2D(in_ch, out_channels, norm_num_groups))

        if add_downsample:
            self.downsamplers = nn.ModuleList([_Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(op, hidden_states)

        return hidden_states


class _Downsample2D(nn.Module):
    """Downsample by stride-2 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return self.conv(op, hidden_states)


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------


class _UpDecoderBlock2D(nn.Module):
    """Up-sampling decoder block: N ResNets + optional upsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(_ResNetBlock2D(in_ch, out_channels, norm_num_groups))

        if add_upsample:
            self.upsamplers = nn.ModuleList([_Upsample2D(out_channels)])
        else:
            self.upsamplers = None

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(op, hidden_states)

        return hidden_states


class _Upsample2D(nn.Module):
    """Upsample by 2x nearest interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = op.Resize(
            hidden_states,
            None,
            op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
            mode="nearest",
        )
        return self.conv(op, hidden_states)


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------


class _MidBlock2D(nn.Module):
    """Mid block: ResNet + Attention + ResNet."""

    def __init__(self, channels: int, norm_num_groups: int = 32, add_attention: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                _ResNetBlock2D(channels, channels, norm_num_groups),
                _ResNetBlock2D(channels, channels, norm_num_groups),
            ]
        )

        if add_attention:
            self.attentions = nn.ModuleList(
                [_AttentionBlock(channels, norm_num_groups=norm_num_groups)]
            )
        else:
            self.attentions = None

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.resnets[0](op, hidden_states)

        if self.attentions is not None:
            hidden_states = self.attentions[0](op, hidden_states)

        hidden_states = self.resnets[1](op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# VAE Encoder and Decoder
# ---------------------------------------------------------------------------


class _VAEEncoder(nn.Module):
    """VAE encoder: image → latent distribution parameters."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        self.conv_in = _Conv2d(
            config.in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            self.down_blocks.append(
                _DownEncoderBlock2D(
                    in_ch,
                    out_ch,
                    num_layers=config.layers_per_block,
                    norm_num_groups=config.norm_num_groups,
                    add_downsample=not is_last,
                )
            )
            in_ch = out_ch

        self.mid_block = _MidBlock2D(
            block_out_channels[-1],
            norm_num_groups=config.norm_num_groups,
            add_attention=config.mid_block_add_attention,
        )

        self.conv_norm_out = _GroupNorm(config.norm_num_groups, block_out_channels[-1])
        # Output 2*latent_channels for mean and logvar
        self.conv_out = _Conv2d(
            block_out_channels[-1], 2 * config.latent_channels, kernel_size=3, padding=1
        )
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, sample: ir.Value):
        hidden_states = self.conv_in(op, sample)

        for down_block in self.down_blocks:
            hidden_states = down_block(op, hidden_states)

        hidden_states = self.mid_block(op, hidden_states)

        hidden_states = self.conv_norm_out(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv_out(op, hidden_states)

        return hidden_states


class _VAEDecoder(nn.Module):
    """VAE decoder: latent → image."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = _Conv2d(
            config.latent_channels, reversed_channels[0], kernel_size=3, padding=1
        )

        self.mid_block = _MidBlock2D(
            reversed_channels[0],
            norm_num_groups=config.norm_num_groups,
            add_attention=config.mid_block_add_attention,
        )

        self.up_blocks = nn.ModuleList()
        in_ch = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            is_last = i == len(reversed_channels) - 1
            self.up_blocks.append(
                _UpDecoderBlock2D(
                    in_ch,
                    out_ch,
                    num_layers=config.layers_per_block + 1,
                    norm_num_groups=config.norm_num_groups,
                    add_upsample=not is_last,
                )
            )
            in_ch = out_ch

        self.conv_norm_out = _GroupNorm(config.norm_num_groups, reversed_channels[-1])
        self.conv_out = _Conv2d(
            reversed_channels[-1], config.out_channels, kernel_size=3, padding=1
        )
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, latent_sample: ir.Value):
        hidden_states = self.conv_in(op, latent_sample)

        hidden_states = self.mid_block(op, hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(op, hidden_states)

        hidden_states = self.conv_norm_out(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv_out(op, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# AutoencoderKL Model
# ---------------------------------------------------------------------------


class AutoencoderKLModel(nn.Module):
    """AutoencoderKL (VAE) model for latent diffusion.

    This model provides both encoder and decoder as a ModelPackage
    with "encoder" and "decoder" components.
    """

    default_task: str = "vae"
    category: str = "autoencoder"

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = _VAEEncoder(config)
        self.decoder = _VAEDecoder(config)

        if config.use_quant_conv:
            self.quant_conv = _Conv2d(
                2 * config.latent_channels,
                2 * config.latent_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.quant_conv = None

        if config.use_post_quant_conv:
            self.post_quant_conv = _Conv2d(
                config.latent_channels,
                config.latent_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.post_quant_conv = None

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
