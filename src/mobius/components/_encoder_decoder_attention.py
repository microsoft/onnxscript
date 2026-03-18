# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoder-decoder attention module for seq2seq models (BART, T5, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Embedding, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class EncoderDecoderAttention(nn.Module):
    """Multi-head attention for encoder-decoder models.

    Supports self-attention and cross-attention (via ``key_value_states``),
    optional relative position bias (T5-style), configurable causality,
    and KV cache for autoregressive decoding.

    Named by pattern, not model: consolidates the identical attention patterns
    used in BART, T5, and similar encoder-decoder architectures.

    Args:
        config: Architecture configuration.
        is_causal: Whether to use causal (unidirectional) attention.
        has_relative_attention_bias: Whether to include T5-style learned
            relative position bias.
        bias: Whether projection layers use bias. Default True (BART-style).
            Pass False for T5-style.
        scale: Attention score scale factor. Defaults to ``1/sqrt(head_dim)``.
            T5 uses ``scale=1.0`` (no scaling).
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        is_causal: bool = False,
        has_relative_attention_bias: bool = False,
        bias: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.is_causal = is_causal
        self._scale = scale if scale is not None else float(self.head_dim**-0.5)

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                config.relative_attention_num_buckets, self.num_heads
            )
        else:
            self.relative_attention_bias = None

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        key_value_states: ir.Value | None = None,
        attention_bias: ir.Value | None = None,
        past_key_value: tuple | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            hidden_states: Query source tensor.
            key_value_states: If provided, K/V are projected from this tensor
                (cross-attention). Otherwise, self-attention is performed.
            attention_bias: Optional attention bias (e.g., relative position bias).
            past_key_value: Cached (key, value) tuple for incremental decoding.

        Returns:
            Tuple of (output, (present_key, present_value)).
        """
        query_states = self.q_proj(op, hidden_states)

        if key_value_states is not None:
            # Cross-attention: K/V projected from encoder hidden states.
            # Encoder output is constant across decode steps, so we
            # recompute the same projections each step and never
            # concatenate with past — concatenation would incorrectly
            # double the cross-attention sequence length.
            key_states = self.k_proj(op, key_value_states)
            value_states = self.v_proj(op, key_value_states)
            past_key = None
            past_value = None
        else:
            key_states = self.k_proj(op, hidden_states)
            value_states = self.v_proj(op, hidden_states)
            if past_key_value is not None:
                past_key, past_value = past_key_value
            else:
                past_key = None
                past_value = None

        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key,
            past_value,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self._scale,
            is_causal=1 if self.is_causal else 0,
            _outputs=3,
        )

        output = self.out_proj(op, attn_output)
        return output, (present_key, present_value)
