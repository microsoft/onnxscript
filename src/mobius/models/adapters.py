# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""T2I-Adapter and IP-Adapter for conditioning image generation.

T2I-Adapter: Lightweight conditioning adapter that produces residuals for UNet.
IP-Adapter: Image Prompt adapter using image embeddings for conditioning.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components import Conv2d as _Conv2d
from mobius.components import GroupNorm as _GroupNorm
from mobius.components import Linear as _Linear
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir


@dataclasses.dataclass
class T2IAdapterConfig:
    """Configuration for T2I-Adapter."""

    in_channels: int = 3
    channels: tuple[int, ...] = (320, 640, 1280, 1280)
    num_res_blocks: int = 2
    adapter_type: str = "full_adapter"

    @classmethod
    def from_diffusers(cls, config: dict) -> T2IAdapterConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 3),
            channels=tuple(config.get("channels", [320, 640, 1280, 1280])),
            num_res_blocks=config.get("num_res_blocks", 2),
        )


class _T2IAdapterBlock(nn.Module):
    """T2I-Adapter conditioning block: Conv + ResNet blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = _Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.resnets = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.resnets.append(_SimpleResBlock(out_channels))

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        if self.downsample is not None:
            hidden_states = self.downsample(op, hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(op, hidden_states)
        return hidden_states


class _SimpleResBlock(nn.Module):
    """Simple ResNet block: Conv → GroupNorm → SiLU → Conv + skip."""

    def __init__(self, channels: int, norm_num_groups: int = 32):
        super().__init__()
        self.norm1 = _GroupNorm(min(norm_num_groups, channels), channels)
        self.conv1 = _Conv2d(channels, channels)
        self.norm2 = _GroupNorm(min(norm_num_groups, channels), channels)
        self.conv2 = _Conv2d(channels, channels)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv1(op, hidden_states)
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv2(op, hidden_states)
        return op.Add(hidden_states, residual)


class T2IAdapterModel(nn.Module):
    """T2I-Adapter: lightweight image conditioning adapter.

    Takes conditioning image and produces multi-scale feature maps to add
    to UNet down blocks during denoising.
    """

    default_task: str = "adapter"
    category: str = "Diffusion"

    def __init__(self, config: T2IAdapterConfig):
        super().__init__()
        self.config = config
        channels = config.channels

        # Pixel unshuffle input conv (4x4 stride to reduce resolution)
        self.unshuffle = _Conv2d(
            config.in_channels * 16, channels[0], kernel_size=3, padding=1
        )

        # Adapter blocks
        self.body = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.body.append(_T2IAdapterBlock(in_ch, out_ch, config.num_res_blocks))

        # Downsampling between blocks
        self.downsamplers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downsamplers.append(
                _Conv2d(channels[i], channels[i], kernel_size=3, stride=2, padding=1)
            )

    def forward(self, op: builder.OpBuilder, condition: ir.Value):
        """Forward pass: conditioning image → multi-scale features.

        Args:
            op: ONNX op builder.
            condition: [batch, channels, height, width] conditioning image

        Returns:
            features: list of feature maps at each scale
        """
        # Pixel unshuffle: [B, 3, H, W] → [B, 48, H/4, W/4]
        hidden_states = op.SpaceToDepth(condition, blocksize=4)
        hidden_states = self.unshuffle(op, hidden_states)

        features = []
        for i, block in enumerate(self.body):
            hidden_states = block(op, hidden_states)
            features.append(hidden_states)
            if i < len(self.downsamplers):
                hidden_states = self.downsamplers[i](op, hidden_states)

        return features

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return state_dict


@dataclasses.dataclass
class IPAdapterConfig:
    """Configuration for IP-Adapter."""

    image_embed_dim: int = 1024
    cross_attention_dim: int = 768
    num_tokens: int = 4

    @classmethod
    def from_diffusers(cls, config: dict) -> IPAdapterConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            image_embed_dim=config.get("image_embed_dim", 1024),
            cross_attention_dim=config.get("cross_attention_dim", 768),
            num_tokens=config.get("num_tokens", 4),
        )


class IPAdapterModel(nn.Module):
    """IP-Adapter: image prompt adapter for cross-attention conditioning.

    Projects CLIP image embeddings to additional KV pairs that are concatenated
    to the text KV pairs in UNet cross-attention layers.
    """

    default_task: str = "adapter"
    category: str = "Diffusion"

    def __init__(self, config: IPAdapterConfig):
        super().__init__()
        self.config = config
        # Image projection: maps image embeddings to cross-attention tokens
        self.image_proj = _ImageProjection(
            config.image_embed_dim,
            config.cross_attention_dim,
            config.num_tokens,
        )

    def forward(self, op: builder.OpBuilder, image_embeds: ir.Value):
        """Project image embeddings to cross-attention tokens.

        Args:
            op: ONNX op builder.
            image_embeds: [batch, image_embed_dim] from CLIP image encoder

        Returns:
            ip_tokens: [batch, num_tokens, cross_attention_dim]
        """
        return self.image_proj(op, image_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return state_dict


class _ImageProjection(nn.Module):
    """Project image embeddings to multiple cross-attention tokens."""

    def __init__(self, image_embed_dim: int, cross_attention_dim: int, num_tokens: int):
        super().__init__()
        self.proj = _Linear(image_embed_dim, cross_attention_dim * num_tokens)
        self.norm = _LayerNorm1D(cross_attention_dim)
        self._num_tokens = num_tokens
        self._cross_dim = cross_attention_dim

    def forward(self, op: builder.OpBuilder, image_embeds: ir.Value):
        # [B, image_dim] → [B, num_tokens * cross_dim] → [B, num_tokens, cross_dim]
        hidden = self.proj(op, image_embeds)
        batch = op.Shape(hidden, start=0, end=1)
        hidden = op.Reshape(
            hidden,
            op.Concat(
                batch, op.Constant(value_ints=[self._num_tokens, self._cross_dim]), axis=0
            ),
        )
        return self.norm(op, hidden)


class _LayerNorm1D(nn.Module):
    """Layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self.bias = nn.Parameter((dim,))
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.LayerNormalization(x, self.weight, self.bias, axis=-1, epsilon=self._eps)
