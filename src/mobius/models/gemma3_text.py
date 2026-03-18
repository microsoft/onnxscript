# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    OffsetRMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma3TextScaledWordEmbedding(Embedding):
    """Embedding with scaling by sqrt(hidden_size)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self.embed_scale)


class Gemma3DecoderLayer(nn.Module):
    """Gemma3 decoder layer with pre/post feedforward layer norms."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = Attention(config, rms_norm_class=OffsetRMSNorm)
        self.mlp = MLP(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class Gemma3TextModel(nn.Module):
    """Gemma3 text model with hybrid attention (global + sliding window)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype

        embed_scale = float(np.float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window

        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

        # Local RoPE for sliding window layers
        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        local_config.rope_type = "default"
        local_config.rope_scaling = None
        self.rotary_emb_local = initialize_rope(local_config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        position_embeddings_dict = {
            "full_attention": self.rotary_emb(op, position_ids),
            "sliding_attention": self.rotary_emb_local(op, position_ids),
        }

        # Use hidden_states for query length when input_ids is None (VL decoder path)
        query_input = input_ids if input_ids is not None else hidden_states
        attention_bias_dict = {
            "full_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                dtype=self._dtype,
            ),
            "sliding_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            ),
        }

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, layer_type, past_kv in zip(self.layers, self.layer_types, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Gemma3CausalLMModel(CausalLMModel):
    """Gemma 3 text model with hybrid attention (global + sliding window), QK-norm, and four-norm decoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Override the model with Gemma3TextModel
        self.model = Gemma3TextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess weights, handling language_model prefix from multimodal."""
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "multi_modal_projector" in key:
                state_dict.pop(key)
        return super().preprocess_weights(state_dict)
