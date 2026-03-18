# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    RMSNorm,
    create_attention_bias,
    create_decoder_layer,
    initialize_rope,
)
from mobius.models.base import CausalLMModel


class _WeightFreeLayerNorm(nn.Module):
    """LayerNorm without learnable parameters (OLMo-style).

    Uses constant scale=1 and bias=0, matching HF OlmoLayerNorm which
    applies F.layer_norm with no weight/bias and eps=1e-5.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(
            [hidden_size], data=ir.tensor(np.ones(hidden_size, dtype=np.float32))
        )
        self.bias = nn.Parameter(
            [hidden_size], data=ir.tensor(np.zeros(hidden_size, dtype=np.float32))
        )
        self.eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return op.LayerNormalization(
            hidden_states, self.scale, self.bias, epsilon=self.eps, axis=-1
        )


class _OlmoDecoderLayer(nn.Module):
    """OLMo decoder layer with weight-free LayerNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = _WeightFreeLayerNorm(config.hidden_size)
        self.post_attention_layernorm = _WeightFreeLayerNorm(config.hidden_size)

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
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class _OlmoTextModel(nn.Module):
    """OLMo text model with weight-free LayerNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_OlmoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _WeightFreeLayerNorm(config.hidden_size)
        self.rotary_emb = initialize_rope(config)

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
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class OLMoCausalLMModel(CausalLMModel):
    """OLMo model with weight-free LayerNorm (not RMSNorm).

    OLMo-1B uses LayerNorm without learnable weight/bias and eps=1e-5,
    unlike Llama which uses RMSNorm with learnable weight.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = _OlmoTextModel(config)


class _PostNormTextModel(nn.Module):
    """Text model with post-norm decoder layers (OLMo-2/OLMo-3 style)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                create_decoder_layer(config, post_norm=True)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

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
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class OLMo2CausalLMModel(CausalLMModel):
    """OLMo-2/OLMo-3 model with post-norm decoder layers and QK normalization."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = _PostNormTextModel(config)
