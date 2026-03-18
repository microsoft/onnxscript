# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    DecoderLayer,
    Embedding,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class SmolLM3TextModel(nn.Module):
    """SmolLM3 text model with per-layer RoPE control and sliding window attention.

    SmolLM3 features:
    - Per-layer conditional RoPE (some layers have no RoPE)
    - Per-layer attention type (sliding_attention vs full_attention)
    - Dynamic window sizing per layer
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        full_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        sliding_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            layer_type = self.layer_types[i] if self.layer_types else "full_attention"
            attn_bias = (
                sliding_attn_bias if layer_type == "sliding_attention" else full_attn_bias
            )

            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class SmolLM3CausalLMModel(CausalLMModel):
    """SmolLM3 causal language model with per-layer attention control."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = SmolLM3TextModel(config)
