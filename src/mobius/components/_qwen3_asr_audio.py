# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR audio encoder components.

Whisper-inspired audio encoder with 3x Conv2d downsampling,
sinusoidal positional embeddings, and bidirectional transformer
encoder layers with LayerNorm.

Reference: QwenLM/Qwen3-ASR modeling_qwen3_asr.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class Qwen3ASRAudioAttention(nn.Module):
    """Bidirectional multi-head attention for Qwen3-ASR audio encoder.

    Unlike WhisperAttention, all projections (Q, K, V, Out) have bias.
    No causal masking — the encoder uses full bidirectional attention.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        """Bidirectional self-attention.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Use ONNX Attention op (bidirectional: no causal mask)
        attn_output = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
        )
        return self.out_proj(op, attn_output)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Qwen3-ASR audio encoder layer.

    Pre-norm pattern: LayerNorm → self-attn → residual
    → LayerNorm → FFN → residual.
    Uses GELU activation in the FFN.

    HuggingFace class: ``Qwen3ASRAudioEncoderLayer``
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = Qwen3ASRAudioAttention(d_model, num_heads)
        self.self_attn_layer_norm = LayerNorm(d_model, eps=eps)
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model, eps=eps)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        """Pre-norm encoder layer with bidirectional attention.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm, GELU, and residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states
