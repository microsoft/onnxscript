# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ControlNet model for conditional image generation.

ControlNet copies the UNet encoder structure and adds zero-conv output layers
to produce residuals that are injected into the UNet during denoising.

Architecture:
1. Same input conv + time embedding as UNet
2. Additional conditioning image input via controlnet_cond_embedding
3. Same down blocks and mid block as UNet
4. Zero-conv outputs at each block level

HF diffusers class: ControlNetModel
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components import Conv2d as _BaseConv2d
from mobius.components import SiLU as _SiLU
from mobius.models.unet import (
    _DownBlock2D,
    _TimestepEmbedding,
    _UNetMidBlock2DCrossAttn,
)

if TYPE_CHECKING:
    import onnx_ir as ir


@dataclasses.dataclass
class ControlNetConfig:
    """Configuration for ControlNet models."""

    in_channels: int = 4
    conditioning_channels: int = 3
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    cross_attention_dim: int = 768
    attention_head_dim: int = 8

    @classmethod
    def from_diffusers(cls, config: dict) -> ControlNetConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 4),
            conditioning_channels=config.get("conditioning_channels", 3),
            block_out_channels=tuple(config.get("block_out_channels", [320, 640, 1280, 1280])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            cross_attention_dim=config.get("cross_attention_dim", 768),
            attention_head_dim=config.get("attention_head_dim", 8),
        )


class _ControlNetConditioningEmbedding(nn.Module):
    """Projects conditioning image to latent space via stacked Conv2d."""

    def __init__(self, conditioning_channels: int, block_out_channels: tuple[int, ...]):
        super().__init__()
        self.conv_in = _BaseConv2d(conditioning_channels, 16, kernel_size=3, padding=1)
        # Intermediate convolutions
        channels = [16, 32, 96, block_out_channels[0]]
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            conv1 = _BaseConv2d(channels[i], channels[i], kernel_size=3, padding=1)
            conv2 = _BaseConv2d(
                channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1
            )
            self.blocks.append(conv1)
            self.blocks.append(conv2)
        self.conv_out = _BaseConv2d(
            block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1
        )
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, conditioning: ir.Value):
        hidden = self.conv_in(op, conditioning)
        hidden = self._silu(op, hidden)
        for block in self.blocks:
            hidden = block(op, hidden)
            hidden = self._silu(op, hidden)
        return self.conv_out(op, hidden)


class ControlNetModel(nn.Module):
    """ControlNet: conditioning adapter for UNet denoisers.

    Takes noisy latent + timestep + text conditioning + conditioning image,
    outputs residuals to inject into the UNet at each block level.
    """

    default_task: str = "controlnet"
    category: str = "Diffusion"

    def __init__(self, config: ControlNetConfig):
        super().__init__()
        self.config = config

        block_out_channels = config.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # Time embedding
        self.time_proj_dim = block_out_channels[0]
        self.time_embedding = _TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Input convolution (same as UNet)
        self.conv_in = _BaseConv2d(
            config.in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # Conditioning embedding
        self.controlnet_cond_embedding = _ControlNetConditioningEmbedding(
            config.conditioning_channels,
            block_out_channels,
        )

        # Down blocks (same as UNet encoder)
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

        # Zero-conv output layers (one per down block output + mid block)
        self.controlnet_down_blocks = nn.ModuleList()
        # Initial conv_in output
        self.controlnet_down_blocks.append(
            _BaseConv2d(block_out_channels[0], block_out_channels[0], kernel_size=1, padding=0)
        )
        for i, ch in enumerate(block_out_channels):
            for _ in range(config.layers_per_block):
                self.controlnet_down_blocks.append(
                    _BaseConv2d(ch, ch, kernel_size=1, padding=0)
                )
            if i < len(block_out_channels) - 1:
                self.controlnet_down_blocks.append(
                    _BaseConv2d(ch, ch, kernel_size=1, padding=0)
                )
        self.controlnet_mid_block = _BaseConv2d(
            block_out_channels[-1],
            block_out_channels[-1],
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
        controlnet_cond: ir.Value,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            sample: Noisy latent [batch, in_channels, height, width]
            timestep: Diffusion timestep [batch]
            encoder_hidden_states: Text encoder output [batch, seq_len, cross_dim]
            controlnet_cond: Conditioning image [batch, conditioning_channels, height*8, width*8]

        Returns:
            down_block_res_samples: List of residuals for UNet down blocks
            mid_block_res_sample: Residual for UNet mid block
        """
        # Time embedding
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        t_emb = op.Concat(op.Cos(args), op.Sin(args), axis=-1)
        emb = self.time_embedding(op, t_emb)

        # Input conv + conditioning
        sample = self.conv_in(op, sample)
        controlnet_cond = self.controlnet_cond_embedding(op, controlnet_cond)
        sample = op.Add(sample, controlnet_cond)

        # Down blocks
        down_block_res_samples = [sample]
        for down_block in self.down_blocks:
            sample, res_samples = down_block(op, sample, emb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)

        # Mid block
        sample = self.mid_block(op, sample, emb, encoder_hidden_states)

        # Apply zero-conv to get residuals
        controlnet_outputs = []
        for i, res_sample in enumerate(down_block_res_samples):
            controlnet_outputs.append(self.controlnet_down_blocks[i](op, res_sample))

        mid_output = self.controlnet_mid_block(op, sample)

        return controlnet_outputs, mid_output

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
