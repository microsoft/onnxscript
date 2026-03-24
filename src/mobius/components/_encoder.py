# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Encoder-only components for BERT-like models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._mlp import FCMLP

if TYPE_CHECKING:
    import onnx_ir as ir


class EncoderAttention(nn.Module):
    """Bidirectional multi-head attention without KV cache or RoPE.

    This is a simpler attention variant for encoder-only models.
    Uses the ONNX Attention op with no causal mask.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        attn_output = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_attention_heads,
            scale=float(self.head_dim**-0.5),
        )

        return self.out_proj(op, attn_output)


class EncoderLayer(nn.Module):
    """Post-norm encoder layer (BERT-style).

    Structure: attn → residual → norm → mlp → residual → norm
    Uses LayerNorm (with bias) and standard MLP (not SwiGLU).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()
        self.self_attn = EncoderAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            bias=bias,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = FCMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=hidden_act,
            bias=bias,
        )
        self.post_mlp_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        # Self-attention with post-norm
        attn_output = self.self_attn(op, hidden_states, attention_mask)
        hidden_states = self.post_attention_layernorm(op, op.Add(hidden_states, attn_output))

        # MLP with post-norm
        mlp_output = self.mlp(op, hidden_states)
        hidden_states = self.post_mlp_layernorm(op, op.Add(hidden_states, mlp_output))

        return hidden_states


class BertEmbeddings(nn.Module):
    """BERT embeddings: word + position + token_type + LayerNorm."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)
        self.layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        token_type_ids: ir.Value,
    ):
        word_embeds = self.word_embeddings(op, input_ids)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.position_embeddings(op, position_ids)
        token_type_embeds = self.token_type_embeddings(op, token_type_ids)

        embeddings = op.Add(op.Add(word_embeds, position_embeds), token_type_embeds)
        return self.layernorm(op, embeddings)
