# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wav2Vec2 encoder-only audio model.

Supports: wav2vec2, hubert, wavlm (all share similar architecture).

Architecture:
1. CNN feature extractor: multiple Conv1d layers with group norm
2. Feature projection: Linear + LayerNorm
3. Transformer encoder: standard self-attention + FFN layers

HF weight naming:
- wav2vec2.feature_extractor.conv_layers.N.conv.weight → feature_extractor.conv_layers.N.conv.weight
- wav2vec2.feature_projection.projection.weight → feature_projection.projection.weight
- wav2vec2.encoder.layers.N.* → encoder.layers.N.*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class _Conv1dFeatureExtractor(nn.Module):
    """CNN feature extractor: extracts features from raw audio waveform."""

    def __init__(self, conv_channels: list[int], conv_kernel_sizes: list[int]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]
            kernel = conv_kernel_sizes[i]
            layer = _ConvLayerBlock(in_ch, out_ch, kernel, use_group_norm=(i == 0))
            self.conv_layers.append(layer)

    def forward(self, op: builder.OpBuilder, input_values: ir.Value):
        # input_values: [batch, time] → [batch, 1, time]
        hidden_states = op.Unsqueeze(input_values, [1])
        for layer in self.conv_layers:
            hidden_states = layer(op, hidden_states)
        # Output: [batch, channels, time'] → transpose to [batch, time', channels]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return hidden_states


class _ConvLayerBlock(nn.Module):
    """Single Conv1d + optional GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_group_norm: bool = False,
    ):
        super().__init__()
        self.conv = nn.Parameter((out_channels, in_channels, kernel_size))
        self.conv_bias = nn.Parameter((out_channels,))
        self.use_group_norm = use_group_norm
        self.out_channels = out_channels
        if use_group_norm:
            self.layer_norm = nn.Parameter((out_channels,))
            self.layer_norm_bias = nn.Parameter((out_channels,))

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = op.Conv(hidden_states, self.conv, self.conv_bias)
        if self.use_group_norm:
            hidden_states = op.GroupNormalization(
                hidden_states,
                self.layer_norm,
                self.layer_norm_bias,
                num_groups=self.out_channels,
            )
        hidden_states = op.Gelu(hidden_states)
        return hidden_states


class _FeatureProjection(nn.Module):
    """Projects CNN features to hidden size with LayerNorm."""

    def __init__(self, conv_dim: int, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = LayerNorm(conv_dim, eps=eps)
        self.projection = Linear(conv_dim, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.layer_norm(op, hidden_states)
        hidden_states = self.projection(op, hidden_states)
        return hidden_states


class _Wav2Vec2EncoderLayer(nn.Module):
    """Standard transformer encoder layer: self-attn + FFN with pre-norm."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float = 1e-5
    ):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_size, eps=eps)
        head_dim = hidden_size // num_heads
        self.attention = _Wav2Vec2Attention(hidden_size, num_heads, head_dim)
        self.feed_forward = FCMLP(hidden_size, intermediate_size, activation="gelu", bias=True)
        self.final_layer_norm = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(op, hidden_states)
        hidden_states = self.attention(op, hidden_states, attention_mask)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _Wav2Vec2Attention(nn.Module):
    """Self-attention for Wav2Vec2 encoder (bidirectional)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)
        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            is_causal=0,
            scale=float(self.head_dim**-0.5),
        )
        return self.out_proj(op, attn_out)


class _Wav2Vec2Encoder(nn.Module):
    """Wrapper matching HF encoder.layers.{i} nesting."""

    def __init__(self, config: ArchitectureConfig, eps: float = 1e-5):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = _Wav2Vec2EncoderLayer(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                eps=eps,
            )
            self.layers.append(layer)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask)
        return hidden_states


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2 encoder-only audio model.

    Architecture: CNN feature extractor → feature projection → transformer encoder.
    Used for ASR (CTC), audio feature extraction, and audio classification.
    """

    default_task: str = "audio-feature-extraction"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # CNN feature extractor
        conv_channels = getattr(
            config, "conv_channels", [1, 512, 512, 512, 512, 512, 512, 512]
        )
        conv_kernel_sizes = getattr(config, "conv_kernel_sizes", [10, 3, 3, 3, 3, 2, 2])
        self.feature_extractor = _Conv1dFeatureExtractor(conv_channels, conv_kernel_sizes)

        # Feature projection
        conv_dim = conv_channels[-1]
        self.feature_projection = _FeatureProjection(
            conv_dim,
            config.hidden_size,
            eps=getattr(config, "layer_norm_eps", 1e-5),
        )

        # Transformer encoder
        self.encoder = _Wav2Vec2Encoder(
            config,
            eps=getattr(config, "layer_norm_eps", 1e-5),
        )

        self.layer_norm = LayerNorm(
            config.hidden_size,
            eps=getattr(config, "layer_norm_eps", 1e-5),
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        """Forward pass: raw audio → hidden states.

        Args:
            op: ONNX op builder.
            input_values: [batch, time] raw audio waveform
            attention_mask: [batch, time] optional mask

        Returns:
            last_hidden_state: [batch, time', hidden_size]
        """
        hidden_states = self.feature_extractor(op, input_values)
        hidden_states = self.feature_projection(op, hidden_states)

        hidden_states = self.encoder(op, hidden_states, attention_mask)

        hidden_states = self.layer_norm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Wav2Vec2 weight names to our names.

        HF prefix: wav2vec2.* → strip it.
        Attribute names are aligned with HF (out_proj, encoder.layers).
        FFN renames: intermediate_dense → up_proj, output_dense → down_proj (FCMLP naming).
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Strip wav2vec2. / hubert. prefix
            for prefix in ("wav2vec2.", "hubert.", "wavlm."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break
            # FFN: intermediate_dense → up_proj, output_dense → down_proj
            new_key = new_key.replace(".intermediate_dense.", ".up_proj.").replace(
                ".output_dense.", ".down_proj."
            )
            new_state_dict[new_key] = value
        return new_state_dict
