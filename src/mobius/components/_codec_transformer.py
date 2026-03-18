# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Transformer components for the Qwen3-TTS codec tokenizer.

Implements the decoder transformer used in ``Qwen3TTSTokenizerV2Decoder``:
a standard Llama-like transformer with RMSNorm, RoPE, SwiGLU MLP, and
per-sublayer LayerScale.

Also implements the encoder transformer variant which uses LayerNorm
(with bias), GELU fc1/fc2 MLP, and no QK norm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._codec_conv import LayerScale
from mobius.components._common import (
    LayerNorm,
    Linear,
)
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import (
    BaseRope,
    _get_cos_sin_cache,
    apply_rotary_pos_emb,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder Transformer (RMSNorm + SwiGLU + LayerScale + RoPE)
# ---------------------------------------------------------------------------


class CodecDecoderTransformerLayer(nn.Module):
    """Single decoder transformer layer with LayerScale.

    Architecture per layer:
        RMSNorm → Attention(RoPE) → LayerScale → + residual
        RMSNorm → SwiGLU MLP → LayerScale → + residual

    HF class: ``Qwen3TTSTokenizerV2DecoderTransformerLayer``.

    Parameters:
        hidden_size: Model dimension (e.g. 512).
        num_heads: Number of attention heads (e.g. 16).
        num_kv_heads: Number of KV heads (e.g. 16).
        head_dim: Per-head dimension (e.g. 64).
        intermediate_size: MLP intermediate dimension.
        rms_norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _CodecDecoderAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.self_attn_layer_scale = LayerScale(hidden_size)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = _SwiGLUMLP(hidden_size, intermediate_size)
        self.mlp_layer_scale = LayerScale(hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
        attention_mask: ir.Value | None = None,
    ):
        """Forward pass.

        Args:
            hidden_states: (B, T, hidden_size).
            position_embeddings: (cos, sin) each (B, T, rotary_dim).
            attention_mask: optional (B, 1, T, T) or similar.

        Returns:
            (B, T, hidden_size).
        """
        # Self-attention sublayer
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, position_embeddings, attention_mask)
        hidden_states = self.self_attn_layer_scale(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # MLP sublayer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.mlp_layer_scale(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class _CodecDecoderAttention(nn.Module):
    """Multi-head attention with RoPE for the codec decoder transformer.

    Uses ``op.RotaryEmbedding`` on Q and K before passing them
    to the ONNX ``Attention`` op (opset 23).

    Parameters:
        hidden_size: Model hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimension.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        qkv_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        self.q_proj = Linear(hidden_size, qkv_dim, bias=False)
        self.k_proj = Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(qkv_dim, hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
        attention_mask: ir.Value | None = None,
    ):
        """Compute multi-head attention with RoPE.

        Args:
            hidden_states: (B, T, hidden_size).
            position_embeddings: (cos, sin) each (B, T, rotary_dim).
            attention_mask: optional.

        Returns:
            (B, T, hidden_size).
        """
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Apply RoPE via RotaryEmbedding op
        q = apply_rotary_pos_emb(
            op,
            x=q,
            position_embeddings=position_embeddings,
            num_heads=self._num_heads,
        )
        k = apply_rotary_pos_emb(
            op,
            x=k,
            position_embeddings=position_embeddings,
            num_heads=self._num_kv_heads,
        )

        scale = float(self._head_dim**-0.5)

        # ONNX Attention op — RoPE already applied
        attn_out = op.Attention(
            q,
            k,
            v,
            attention_mask,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_kv_heads,
            scale=scale,
        )

        return self.o_proj(op, attn_out)


class _SwiGLUMLP(nn.Module):
    """SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x) → down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        gate = self.gate_proj(op, x)
        up = self.up_proj(op, x)
        # SiLU(gate) * up
        activated = op.Mul(op.Mul(gate, op.Sigmoid(gate)), up)
        return self.down_proj(op, activated)


class CodecDecoderTransformerModel(nn.Module):
    """Full decoder transformer: input_proj → N layers → norm → output_proj.

    Holds the RoPE cos/sin cache and converts position_ids to
    position_embeddings before passing them to each layer.

    HF class: ``Qwen3TTSTokenizerV2DecoderTransformerModel``.

    Parameters:
        latent_dim: Input/output dimension (e.g. 1024).
        hidden_size: Transformer hidden dimension (e.g. 512).
        num_hidden_layers: Number of transformer layers (e.g. 8).
        num_attention_heads: Number of attention heads (e.g. 16).
        num_key_value_heads: Number of KV heads (e.g. 16).
        intermediate_size: MLP intermediate dimension (e.g. 1024).
        head_dim: Per-head dimension (e.g. 64).
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        max_position_embeddings: Max sequence length for RoPE cache.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.input_proj = Linear(latent_dim, hidden_size, bias=True)
        self.layers = nn.ModuleList(
            [
                CodecDecoderTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.output_proj = Linear(hidden_size, latent_dim, bias=True)

        # RoPE cos/sin cache — inv_freq spans head_dim/2 frequencies
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        cos_cache, sin_cache = _get_cos_sin_cache(max_position_embeddings, inv_freq)
        self.rotary_emb = BaseRope(cos_cache, sin_cache)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        """Transform latent representations.

        Args:
            hidden_states: (B, T, latent_dim).
            position_ids: (B, T) int64.
            attention_mask: optional.

        Returns:
            (B, T, latent_dim).
        """
        # Project into transformer hidden space
        hidden_states = self.input_proj(op, hidden_states)

        # Convert position_ids → (cos, sin) embeddings
        position_embeddings = self.rotary_emb(op, position_ids)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, position_embeddings, attention_mask)

        hidden_states = self.norm(op, hidden_states)
        hidden_states = self.output_proj(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder Transformer (LayerNorm + GELU fc1/fc2 + LayerScale)
# ---------------------------------------------------------------------------


class CodecEncoderTransformerLayer(nn.Module):
    """Single encoder transformer layer (Mimi-style).

    Architecture per layer:
        LayerNorm(bias) → Attention(RoPE) → LayerScale → + residual
        LayerNorm(bias) → fc1 → GELU → fc2 → LayerScale → + residual

    Differs from the decoder variant:
    - LayerNorm with bias (not RMSNorm)
    - fc1/fc2 MLP with GELU (not SwiGLU gate/up/down)

    Parameters:
        hidden_size: Model dimension (e.g. 512).
        num_heads: Number of attention heads (e.g. 8).
        num_kv_heads: Number of KV heads (e.g. 8).
        head_dim: Per-head dimension (e.g. 64).
        intermediate_size: MLP intermediate dimension (e.g. 2048).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attn = _CodecDecoderAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.self_attn_layer_scale = LayerScale(hidden_size)

        self.post_attention_layernorm = LayerNorm(hidden_size)
        self.mlp = _GELUFc1Fc2MLP(hidden_size, intermediate_size)
        self.mlp_layer_scale = LayerScale(hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
        attention_mask: ir.Value | None = None,
    ):
        """Forward pass.

        Args:
            hidden_states: (B, T, hidden_size).
            position_embeddings: (cos, sin) each (B, T, rotary_dim).
            attention_mask: optional.

        Returns:
            (B, T, hidden_size).
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, position_embeddings, attention_mask)
        hidden_states = self.self_attn_layer_scale(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.mlp_layer_scale(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class _GELUFc1Fc2MLP(nn.Module):
    """Simple MLP: fc1 → GELU → fc2."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return self.fc2(op, op.Gelu(self.fc1(op, x)))


class CodecEncoderTransformerModel(nn.Module):
    """Full encoder transformer for Mimi-style encoder.

    Holds the RoPE cos/sin cache and converts position_ids to
    position_embeddings before passing them to each layer.

    No input/output projections — the encoder operates at the same
    hidden dimension throughout.

    Parameters:
        hidden_size: Transformer hidden dimension (e.g. 512).
        num_hidden_layers: Number of transformer layers (e.g. 8).
        num_attention_heads: Number of attention heads (e.g. 8).
        num_key_value_heads: Number of KV heads (e.g. 8).
        intermediate_size: MLP intermediate dim (e.g. 2048).
        head_dim: Per-head dimension (e.g. 64).
        rope_theta: RoPE base frequency.
        max_position_embeddings: Max sequence length for RoPE cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        head_dim: int = 64,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CodecEncoderTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # RoPE cos/sin cache
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        cos_cache, sin_cache = _get_cos_sin_cache(max_position_embeddings, inv_freq)
        self.rotary_emb = BaseRope(cos_cache, sin_cache)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        """Transform encoder representations.

        Args:
            hidden_states: (B, T, hidden_size).
            position_ids: (B, T) int64.
            attention_mask: optional.

        Returns:
            (B, T, hidden_size).
        """
        # Convert position_ids → (cos, sin) embeddings
        position_embeddings = self.rotary_emb(op, position_ids)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, position_embeddings, attention_mask)
        return hidden_states
