# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Whisper encoder-decoder components.

Provides modules for the Whisper speech-to-text model:
- ``Conv1d``: 1D convolution layer
- ``WhisperAttention``: Multi-head attention (no RoPE) for self/cross-attention
- ``WhisperEncoderLayer``: Pre-norm encoder layer (self-attention + FFN)
- ``WhisperDecoderLayer``: Pre-norm decoder layer (self-attention + cross-attention + FFN)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class Conv1d(nn.Module):
    """1D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels, kernel_size])
        self.bias = nn.Parameter([out_channels])
        self._kernel_shape = [kernel_size]
        self._strides = [stride]
        self._pads = [padding, padding]

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [batch, in_channels, seq_len]
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
            pads=self._pads,
        )


class WhisperAttention(nn.Module):
    """Multi-head attention for Whisper (no RoPE, with bias).

    Supports self-attention (encoder/decoder) and cross-attention (decoder).
    When ``key_value_states`` is provided, K/V are projected from that source
    instead of from ``hidden_states`` (cross-attention mode).

    For decoder self-attention, set ``is_causal=True`` to use the built-in
    causal mask in the ONNX Attention op instead of an explicit bias.
    """

    def __init__(self, d_model: int, num_heads: int, is_causal: bool = False):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._scale = float(self._head_dim) ** -0.5
        self._is_causal = is_causal
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        key_value_states: ir.Value | None = None,
        past_key_value: tuple | None = None,
    ):
        # Pre-scale Q before attention (matches HF Whisper's scaling order,
        # which is important for numerical parity — see HF comment on
        # "floating point arithmetics' inprecisions").
        q = op.Mul(self.q_proj(op, hidden_states), self._scale)
        kv_source = key_value_states if key_value_states is not None else hidden_states
        k = self.k_proj(op, kv_source)
        v = self.v_proj(op, kv_source)

        # For cross-attention (key_value_states provided), don't
        # concatenate with past — encoder output is constant across
        # decode steps.
        if key_value_states is not None:
            past_k = None
            past_v = None
        elif past_key_value is not None:
            past_k = past_key_value[0]
            past_v = past_key_value[1]
        else:
            past_k = None
            past_v = None

        attn_output, present_key, present_value = op.Attention(
            q,
            k,
            v,
            None,
            past_k,
            past_v,
            kv_num_heads=self._num_heads,
            q_num_heads=self._num_heads,
            scale=1.0,
            is_causal=1 if self._is_causal else 0,
            _outputs=3,
        )
        return self.out_proj(op, attn_output), (present_key, present_value)


class WhisperEncoderLayer(nn.Module):
    """Pre-norm Whisper encoder layer.

    Structure: LayerNorm → SelfAttn → Residual → LayerNorm → FFN → Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        activation: str = "gelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = WhisperAttention(d_model, num_heads)
        self.self_attn_layer_norm = LayerNorm(d_model, eps=eps)
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model, eps=eps)
        self._activation = activation

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, _ = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class WhisperDecoderLayer(nn.Module):
    """Pre-norm Whisper decoder layer with self-attention and cross-attention.

    Structure:
        LayerNorm → SelfAttn → Residual
        → LayerNorm → CrossAttn → Residual
        → LayerNorm → FFN → Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        activation: str = "gelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = WhisperAttention(d_model, num_heads, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(d_model, eps=eps)
        self.encoder_attn = WhisperAttention(d_model, num_heads, is_causal=False)
        self.encoder_attn_layer_norm = LayerNorm(d_model, eps=eps)
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model, eps=eps)
        self._activation = activation

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_value: tuple | None = None,
    ):
        # Self-attention (causal via is_causal attr, with KV cache)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Cross-attention (bidirectional, K/V from encoder)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(op, hidden_states)
        hidden_states, _ = self.encoder_attn(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv
