# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Q-Former (Querying Transformer) component for BLIP-2 style VLMs.

Q-Former uses learned query tokens that cross-attend to visual features,
producing a fixed-size output regardless of input image resolution.
Unlike LLaVA-style MLP projectors, Q-Former is a full transformer encoder
with both self-attention (between queries) and cross-attention (from queries
to visual features).

Architecture per layer:
    Self-Attention → residual + LayerNorm →
    Cross-Attention → residual + LayerNorm →
    FFN → residual + LayerNorm

Reference: BLIP-2 (https://arxiv.org/abs/2301.12597)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._activations import ACT2FN
from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class QFormerAttention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    When ``key_value_states`` is provided to ``forward()``, performs
    cross-attention (queries attend to encoder features). Otherwise
    performs self-attention (queries attend to each other).

    Args:
        hidden_size: Dimension of query hidden states.
        num_attention_heads: Number of attention heads.
        kv_hidden_size: Dimension of key/value source. Defaults to
            ``hidden_size`` (self-attention). Set to encoder hidden size
            for cross-attention when dimensions differ.
        bias: Whether projection layers include bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_hidden_size: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        if kv_hidden_size is None:
            kv_hidden_size = hidden_size

        self.q_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = Linear(kv_hidden_size, hidden_size, bias=bias)
        self.v_proj = Linear(kv_hidden_size, hidden_size, bias=bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        key_value_states: ir.Value | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            hidden_states: Query source. (batch, seq_len, hidden_size)
            key_value_states: If provided, K/V are projected from this
                tensor (cross-attention). Otherwise self-attention.
                (batch, src_len, kv_hidden_size)

        Returns:
            Attention output. (batch, seq_len, hidden_size)
        """
        query_states = self.q_proj(op, hidden_states)

        if key_value_states is not None:
            # Cross-attention: K/V from encoder output
            key_states = self.k_proj(op, key_value_states)
            value_states = self.v_proj(op, key_value_states)
        else:
            # Self-attention: K/V from the same input as Q
            key_states = self.k_proj(op, hidden_states)
            value_states = self.v_proj(op, hidden_states)

        attn_output = op.Attention(
            query_states,
            key_states,
            value_states,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_attention_heads,
            scale=float(self.head_dim**-0.5),
        )

        return self.out_proj(op, attn_output)


class _QFormerMLP(nn.Module):
    """Standard MLP for Q-Former layers (not SwiGLU).

    Structure: Linear(hidden→intermediate) → activation → Linear(intermediate→hidden)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)
        self._act_fn = ACT2FN[hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.up_proj(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        return self.down_proj(op, hidden_states)


class QFormerLayer(nn.Module):
    """Single Q-Former transformer layer.

    Post-norm (BERT-style) with three sub-layers:
        1. Self-attention between query tokens
        2. Cross-attention from queries to encoder (visual) features
        3. Feed-forward network

    Each sub-layer uses residual connection followed by LayerNorm.

    Args:
        hidden_size: Hidden dimension of query tokens.
        num_attention_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        encoder_hidden_size: Hidden dimension of encoder features.
            Defaults to ``hidden_size`` if encoder and Q-Former share
            the same dimension.
        hidden_act: Activation function name for FFN.
        layer_norm_eps: Epsilon for LayerNorm.
        bias: Whether layers include bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        encoder_hidden_size: int | None = None,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()
        # Self-attention: queries attend to each other
        self.self_attn = QFormerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            bias=bias,
        )
        self.self_attn_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

        # Cross-attention: queries attend to encoder (visual) features
        self.cross_attn = QFormerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_hidden_size=encoder_hidden_size,
            bias=bias,
        )
        self.cross_attn_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

        # Feed-forward network
        self.mlp = _QFormerMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=bias,
        )
        self.mlp_layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            hidden_states: Query token states. (batch, num_queries, hidden_size)
            encoder_hidden_states: Visual features from encoder.
                (batch, num_patches, encoder_hidden_size)

        Returns:
            Updated query states. (batch, num_queries, hidden_size)
        """
        # Self-attention with post-norm
        self_attn_output = self.self_attn(op, hidden_states)
        hidden_states = self.self_attn_layernorm(op, op.Add(hidden_states, self_attn_output))

        # Cross-attention with post-norm
        cross_attn_output = self.cross_attn(
            op, hidden_states, key_value_states=encoder_hidden_states
        )
        hidden_states = self.cross_attn_layernorm(op, op.Add(hidden_states, cross_attn_output))

        # FFN with post-norm
        mlp_output = self.mlp(op, hidden_states)
        hidden_states = self.mlp_layernorm(op, op.Add(hidden_states, mlp_output))

        return hidden_states


class QFormer(nn.Module):
    """Querying Transformer for BLIP-2 style vision-language bridging.

    Maintains a set of learned query tokens that cross-attend to visual
    features from a frozen vision encoder. Produces a fixed-size output
    ``(batch, num_query_tokens, hidden_size)`` regardless of input image
    resolution or number of visual patches.

    Args:
        num_query_tokens: Number of learned query tokens (typically 32).
        num_layers: Number of Q-Former transformer layers.
        hidden_size: Hidden dimension of query tokens.
        num_attention_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        encoder_hidden_size: Hidden dimension of visual features from the
            vision encoder. Defaults to ``hidden_size``.
        hidden_act: Activation function for FFN layers.
        layer_norm_eps: Epsilon for LayerNorm layers.
        bias: Whether layers include bias terms.
    """

    def __init__(
        self,
        num_query_tokens: int,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        encoder_hidden_size: int | None = None,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        # Learned query embeddings: (num_query_tokens, hidden_size)
        self.query_tokens = nn.Parameter([num_query_tokens, hidden_size])

        # Stack of Q-Former transformer layers
        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    encoder_hidden_size=encoder_hidden_size,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.layernorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        encoder_hidden_states: ir.Value,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            encoder_hidden_states: Visual features from the vision encoder.
                (batch, num_patches, encoder_hidden_size)

        Returns:
            Query output. (batch, num_query_tokens, hidden_size)
        """
        # Expand query tokens to batch size:
        # (num_queries, hidden) → (1, num_queries, hidden)
        query_embeds = op.Unsqueeze(self.query_tokens, [0])
        # Broadcast to (batch, num_queries, hidden) by expanding
        # along the batch dimension to match encoder input
        batch_size = op.Shape(encoder_hidden_states, start=0, end=1)
        num_queries = op.Constant(value_int=self.num_query_tokens)
        hidden_dim = op.Shape(query_embeds, start=2, end=3)
        target_shape = op.Concat(batch_size, op.Reshape(num_queries, [-1]), hidden_dim, axis=0)
        hidden_states = op.Expand(query_embeds, target_shape)

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, encoder_hidden_states)

        # Final layer norm
        hidden_states = self.layernorm(op, hidden_states)

        return hidden_states
