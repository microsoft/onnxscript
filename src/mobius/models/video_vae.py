# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Video autoencoder model for 3D (temporal) diffusion.

Extends the 2D VAE with temporal convolutions for video encoding/decoding.
Used by CogVideoX, Open-Sora, and similar video generation models.

Architecture:
- 3D convolutions (spatial + temporal)
- Causal temporal attention
- Down/up sampling in both spatial and temporal dimensions
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components import GroupNorm as _GroupNorm
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir


@dataclasses.dataclass
class VideoVAEConfig:
    """Configuration for 3D video autoencoder."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    temporal_compression_ratio: int = 4
    sample_size: int = 64

    @classmethod
    def from_diffusers(cls, config: dict) -> VideoVAEConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 4),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            temporal_compression_ratio=config.get("temporal_compression_ratio", 4),
        )


# ---------------------------------------------------------------------------
# 3D convolution building blocks
# ---------------------------------------------------------------------------


class _Conv3d(nn.Module):
    """3D convolution with bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        self.weight = nn.Parameter((out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = list(kernel_size)
        self._stride = list(stride)
        self._padding = list(padding)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        pads = [
            p for p in self._padding for _ in range(2)
        ]  # [d_begin, h_begin, w_begin, d_end, h_end, w_end]
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_size,
            strides=self._stride,
            pads=pads,
        )


class _ResNetBlock3D(nn.Module):
    """3D ResNet block with GroupNorm → SiLU → Conv3d + skip."""

    def __init__(self, in_channels: int, out_channels: int, norm_num_groups: int = 32):
        super().__init__()
        self.norm1 = _GroupNorm(norm_num_groups, in_channels)
        self.conv1 = _Conv3d(in_channels, out_channels)
        self.norm2 = _GroupNorm(norm_num_groups, out_channels)
        self.conv2 = _Conv3d(out_channels, out_channels)
        self._silu = _SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = _Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            )
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


class _DownBlock3D(nn.Module):
    """3D down-sampling block: N ResNets + spatial downsample."""

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
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(_ResNetBlock3D(res_in, out_channels, norm_num_groups))

        self.downsamplers = None
        if add_downsample:
            # Downsample spatially (stride 2 in H,W), keep temporal
            self.downsamplers = nn.ModuleList()
            self.downsamplers.append(
                _Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 3, 3),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                )
            )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(op, hidden_states)
        return hidden_states


class _UpBlock3D(nn.Module):
    """3D up-sampling block: N ResNets + spatial upsample."""

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
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(_ResNetBlock3D(res_in, out_channels, norm_num_groups))

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList()
            self.upsamplers.append(
                _Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)
        if self.upsamplers is not None:
            # Upsample 2x spatially via nearest neighbor
            hidden_states = op.Resize(
                hidden_states,
                None,
                None,
                op.Constant(value_floats=[1.0, 1.0, 1.0, 2.0, 2.0]),
                mode="nearest",
            )
            for conv in self.upsamplers:
                hidden_states = conv(op, hidden_states)
        return hidden_states


class _MidBlock3D(nn.Module):
    """3D mid block: ResNet + ResNet (no spatial attention for simplicity)."""

    def __init__(self, channels: int, norm_num_groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.resnets.append(_ResNetBlock3D(channels, channels, norm_num_groups))
        self.resnets.append(_ResNetBlock3D(channels, channels, norm_num_groups))

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Video Encoder / Decoder
# ---------------------------------------------------------------------------


class _VideoEncoder(nn.Module):
    """3D video encoder: conv_in → down_blocks → mid_block → norm → conv_out."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        self.conv_in = _Conv3d(config.in_channels, block_out_channels[0])

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = ch
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(
                _DownBlock3D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block,
                    norm_num_groups=config.norm_num_groups,
                    add_downsample=not is_final,
                )
            )

        self.mid_block = _MidBlock3D(block_out_channels[-1], config.norm_num_groups)
        self.conv_norm_out = _GroupNorm(config.norm_num_groups, block_out_channels[-1])
        # Encode to 2*latent_channels (mean + logvar)
        self.conv_out = _Conv3d(
            block_out_channels[-1],
            2 * config.latent_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
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


class _VideoDecoder(nn.Module):
    """3D video decoder: conv_in → mid_block → up_blocks → norm → conv_out."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        block_out_channels = config.block_out_channels
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = _Conv3d(config.latent_channels, reversed_channels[0])

        self.mid_block = _MidBlock3D(reversed_channels[0], config.norm_num_groups)

        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = ch
            is_final = i == len(reversed_channels) - 1
            self.up_blocks.append(
                _UpBlock3D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block + 1,
                    norm_num_groups=config.norm_num_groups,
                    add_upsample=not is_final,
                )
            )

        self.conv_norm_out = _GroupNorm(config.norm_num_groups, block_out_channels[0])
        self.conv_out = _Conv3d(
            block_out_channels[0],
            config.out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
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
# Full Video Autoencoder
# ---------------------------------------------------------------------------


class VideoAutoencoderModel(nn.Module):
    """Video autoencoder with 3D convolutions for temporal + spatial encoding.

    Used by CogVideoX, Open-Sora, and similar video generation models.
    Encodes video [B, C, T, H, W] to latent [B, C', T', H', W'] and decodes back.
    """

    default_task: str = "vae"
    category: str = "Diffusion"

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = _VideoEncoder(config)
        self.decoder = _VideoDecoder(config)
        self.quant_conv = None
        self.post_quant_conv = None

    def forward(self, op: builder.OpBuilder, latent_sample: ir.Value):
        """Decoder forward pass."""
        return self.decoder(op, latent_sample)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return state_dict
