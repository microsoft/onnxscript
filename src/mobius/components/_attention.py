# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import OffsetRMSNorm, RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb

if TYPE_CHECKING:
    import onnx_ir as ir


class Attention(nn.Module):
    """Multi-head attention module using ONNX ops.

    Supports GQA (grouped query attention), optional Q/K normalization,
    and optional rotary position embeddings.

    Args:
        config: Architecture configuration.
        rms_norm_class: Norm class for Q/K normalization (default: RMSNorm).
        scale: Custom attention scale factor (default: 1/sqrt(head_dim)).
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``. Pass a
            ``LoRALinear`` factory for LoRA-adapted attention.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        rms_norm_class: type[nn.Module] | None = None,
        scale: float | None = None,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = scale if scale is not None else self.head_dim**-0.5
        self.rotary_embedding_dim = (
            0
            if math.isclose(config.partial_rotary_factor, 1.0)
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave

        self.q_proj = linear_class(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = linear_class(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attn_o_bias,
        )

        if config.attn_qk_norm:
            rms_norm_class = RMSNorm if rms_norm_class is None else rms_norm_class
            self._qk_norm_full = config.attn_qk_norm_full
            if self._qk_norm_full:
                self.q_norm = rms_norm_class(
                    self.num_attention_heads * self.head_dim, eps=config.rms_norm_eps
                )
                self.k_norm = rms_norm_class(
                    self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps
                )
            else:
                self.q_norm = rms_norm_class(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = rms_norm_class(self.head_dim, eps=config.rms_norm_eps)
        else:
            self._qk_norm_full = False
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        if self.q_norm is not None and self.k_norm is not None:
            if self._qk_norm_full:
                # Apply norm on 3D tensor (across all heads)
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                # Apply norm per-head on 4D tensor
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

        # Apply rotary position embeddings (skip when not provided)
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        # Use ONNX Attention op (opset 23)
        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            kv_num_heads=self.num_key_value_heads,
            q_num_heads=self.num_attention_heads,
            scale=self.scaling,
            _outputs=3,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class Qwen35Attention(nn.Module):
    """Multi-head attention with output gating for Qwen3.5.

    Differences from base Attention:
    - Q projection is doubled to produce both Q and a gating signal
    - Per-head Q/K RMSNorm with +1 offset (OffsetRMSNorm)
    - Partial RoPE (rotary_embedding_dim < head_dim)
    - Output gating: attn_output * sigmoid(gate)
    """

    def __init__(
        self,
        config: ArchitectureConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_embedding_dim = (
            0
            if math.isclose(config.partial_rotary_factor, 1.0)
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave

        q_dim = self.num_attention_heads * self.head_dim
        self.q_proj = Linear(
            self.hidden_size,
            q_dim * 2,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            q_dim,
            self.hidden_size,
            bias=config.attn_o_bias,
        )

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Q projection (doubled) → split into Q and gate per head
        q_gate = self.q_proj(op, hidden_states)
        # Reshape to per-head view so split separates Q/gate within each head
        q_gate = op.Reshape(
            q_gate,
            [0, 0, self.num_attention_heads, self.head_dim * 2],
        )
        query_states, gate = op.Split(q_gate, num_outputs=2, axis=-1, _outputs=2)
        gate = op.Reshape(gate, [0, 0, -1])

        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Per-head RMSNorm on 4D tensors (query_states already 4D)
        key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        # Apply rotary position embeddings
        query_states = apply_rotary_pos_emb(
            op,
            x=query_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )
        key_states = apply_rotary_pos_emb(
            op,
            x=key_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_key_value_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        # Use ONNX Attention op (opset 23)
        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            kv_num_heads=self.num_key_value_heads,
            q_num_heads=self.num_attention_heads,
            scale=self.scaling,
            _outputs=3,
        )

        # Output gating: attn_output * sigmoid(gate)
        attn_output = op.Mul(attn_output, op.Sigmoid(gate))

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)
