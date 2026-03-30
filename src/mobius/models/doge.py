# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DOGE causal language model.

Extends standard attention with a dynamic SSM-style attention mask and
learnable residual gates.

DOGE attention computes a per-position decay from the value states:
    dt_states = exp(A * softplus(dt_proj(V_reshaped)))
    attn_mask[i,j] = dt_states[:, :, j]   (broadcast across query dim)
The dynamic mask REPLACES causal masking entirely — there is no hard
causal mask. Causal behavior emerges from the learned A decay parameter.

Decoder layers use learnable element-wise residual scaling:
    hidden = input_residual * residual + attn_output

Replicates HuggingFace ``DogeForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._attention import Attention, StaticCacheState
from mobius.components._common import (
    Embedding,
    Linear,
    create_padding_mask,
)
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb, initialize_rope
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class DogeAttention(Attention):
    """DOGE attention with SSM-style dynamic attention mask.

    Standard Q/K/V projections + QK-norm + RoPE, plus:
    - A: learnable SSM decay parameter [num_kv_heads]
    - dt_proj: Linear(kv_dim → num_kv_heads) for decay time constant
    - Dynamic mask: exp(A * softplus(dt_proj(V))) broadcast as attention bias
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int = 0):
        super().__init__(config)
        kv_dim = config.num_key_value_heads * (
            config.head_dim or config.hidden_size // config.num_attention_heads
        )
        # SSM parameters
        self.A = nn.Parameter([config.num_key_value_heads])
        self.dt_proj = Linear(kv_dim, config.num_key_value_heads, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        # Standard Q/K/V projections (from base Attention)
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # QK-norm (per-head): reshape 3D→4D, norm, reshape back
        if self.q_norm is not None and self.k_norm is not None:
            query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
            key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
            query_states = self.q_norm(op, query_states)
            key_states = self.k_norm(op, key_states)
            query_states = op.Reshape(query_states, [0, 0, -1])
            key_states = op.Reshape(key_states, [0, 0, -1])

        # RoPE
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

        # --- DOGE dynamic mask from value states ---
        # DOGE does NOT use a causal mask. The dynamic mask (dt_states)
        # IS the complete attention mask. Causal behavior emerges from
        # learned A decay parameters (earlier KV positions decay more).
        # dt_proj: [B, S, kv_dim] → [B, S, kv_heads]
        dt_states = self.dt_proj(op, value_states)
        dt_states = op.Softplus(dt_states)
        dt_states = op.Mul(self.A, dt_states)
        dt_states = op.Exp(dt_states)  # [B, S, kv_heads]
        dt_states = op.Transpose(dt_states, perm=[0, 2, 1])  # [B, kv, S]
        dt_states = op.Unsqueeze(dt_states, [2])  # [B, kv, 1, S]

        # Apply padding: where padding mask is False, set to min_dtype
        # attention_bias is a bool padding mask [B, 1, 1, total_S]
        if attention_bias is not None:
            min_val = op.CastLike(-3.4028235e38, dt_states)
            doge_mask = op.Where(attention_bias, dt_states, min_val)
        else:
            doge_mask = dt_states

        # op.Attention with is_causal=0 and float mask
        past_key = past_key_value[0] if past_key_value is not None else None
        past_value = past_key_value[1] if past_key_value is not None else None
        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            doge_mask,
            past_key,
            past_value,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            is_causal=0,
            _outputs=3,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class DogeDecoderLayer(nn.Module):
    """DOGE decoder layer with learnable residual gates.

    Uses element-wise learnable scaling: hidden = residual * gate + output
    instead of standard residual addition.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DogeAttention(config, layer_idx=layer_idx)
        self.input_residual = nn.Parameter([config.hidden_size])

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = self._make_mlp(config)
        self.post_attention_residual = nn.Parameter([config.hidden_size])

        self._rope = initialize_rope(config)

    @staticmethod
    def _make_mlp(config: ArchitectureConfig):
        from mobius.components._mlp import MLP

        return MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_ids: ir.Value | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        position_embeddings = self._rope(op, position_ids)

        # Attention with learnable residual gate
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )
        # Learnable residual: input_residual * residual + attn_output
        hidden_states = op.Add(op.Mul(self.input_residual, residual), hidden_states)

        # MLP with learnable residual gate
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(op.Mul(self.post_attention_residual, residual), hidden_states)

        return hidden_states, present_key_value


class DogeTextModel(nn.Module):
    """DOGE text model with custom decoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DogeDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Bool padding mask only — DOGE does NOT use causal masking.
        # The dynamic mask (dt_states) replaces causal attention.
        if attention_mask is not None:
            padding_mask = create_padding_mask(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            padding_mask = None

        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = layer(
                op,
                hidden_states,
                attention_bias=padding_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                static_cache=static_cache,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class DogeCausalLMModel(CausalLMModel):
    """DOGE model with SSM dynamic attention mask and learnable residual gates."""

    def __init__(self, config: ArchitectureConfig):
        # Enable QK-norm (DOGE uses per-head RMSNorm on Q/K)
        config = dataclasses.replace(config, attn_qk_norm=True)
        # Skip base __init__ — we build our own text model
        nn.Module.__init__(self)
        self.config = config
        self.model = DogeTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            static_cache,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Handle DOGE-specific weight names.

        DOGE's A parameter is a 1D tensor [num_kv_heads], matching
        our nn.Parameter shape. No renaming needed for standard
        Q/K/V/O, norms, MLP — all names match.
        """
        return super().preprocess_weights(state_dict)
