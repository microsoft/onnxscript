# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Base causal language model for standard decoder-only transformers.

Provides TextModel (embedding + decoder layers + norm) and CausalLMModel
(TextModel + LM head). Directly used by Llama, Qwen2, Mistral, and other
architectures that follow the standard GQA + RoPE pattern.

Replicates HuggingFace's LlamaForCausalLM / MistralForCausalLM /
Qwen2ForCausalLM structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, CausalLMConfig
from mobius._weight_utils import (
    preprocess_awq_weights,
    preprocess_gptq_weights,
    tie_word_embeddings,
)
from mobius.components import (
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    create_padding_mask,
    initialize_rope,
    make_quantized_linear_factory,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class TextModel(nn.Module):
    """Base text model with embedding, decoder layers, and final norm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype

        # If the config has quantization, swap Linear for QuantizedLinear
        # in all decoder layer projections (Attention Q/K/V/O + MLP).
        linear_class = None
        qc = getattr(config, "quantization", None)
        if qc is not None and qc.quant_method != "none":
            linear_class = make_quantized_linear_factory(
                bits=qc.bits,
                block_size=qc.group_size,
                has_zero_point=not qc.sym,
            )

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=linear_class)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        # When attention_mask is None (static cache mode), skip mask
        # creation entirely — the Attention op uses is_causal=1 instead.
        # When present, create a bool padding mask. Causal masking is
        # handled by is_causal=1 on the Attention op (set in
        # _apply_attention), so we only need padding information here.
        if attention_mask is not None:
            padding_mask = create_padding_mask(
                op,
                input_ids=hidden_states if input_ids is None else input_ids,
                attention_mask=attention_mask,
            )
        else:
            padding_mask = None

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=padding_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class CausalLMModel(nn.Module):
    """Standard causal language model with TextModel backbone and LM head.

    Compatible with Llama 2/3, Mistral, Qwen2/2.5, and other architectures
    that follow the standard decoder-only transformer pattern with GQA and RoPE.

    Replicates HuggingFace's ``LlamaForCausalLM``, ``MistralForCausalLM``,
    ``Qwen2ForCausalLM``, etc.

    Inputs: input_ids, attention_mask, position_ids, past_key_values.
    Outputs: logits (batch, seq_len, vocab_size), present_key_values.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"
    config_class: type = CausalLMConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess the state_dict to match the model's expected keys."""
        qc = getattr(self.config, "quantization", None)
        if qc is not None and qc.quant_method == "gptq":
            state_dict = preprocess_gptq_weights(
                state_dict, bits=qc.bits, group_size=qc.group_size
            )
        elif qc is not None and qc.quant_method == "awq":
            state_dict = preprocess_awq_weights(
                state_dict, bits=qc.bits, group_size=qc.group_size
            )
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)
        return state_dict
