# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""InternLM2 CausalLM model with HF-aligned weight naming.

InternLM2 uses non-standard HF naming for layers:
- attention_norm / ffn_norm (instead of input_layernorm / post_attention_layernorm)
- feed_forward with w1/w2/w3 (instead of mlp with gate/down/up_proj)
- attention with wo (instead of self_attn with o_proj)
- tok_embeddings (instead of embed_tokens)
- wqkv for fused QKV (grouped/interleaved layout)

Module attribute names here match HF conventions to minimize
preprocess_weights renames. Only tok_embeddings→embed_tokens,
wo→o_proj, and the wqkv split remain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._activations import get_activation

if TYPE_CHECKING:
    import onnx_ir as ir


class _InternLMMLP(nn.Module):
    """SwiGLU MLP with HF InternLM naming: w1=gate, w2=down, w3=up."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.w1 = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.w2 = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.w3 = Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self._act_fn = get_activation(config.hidden_act)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate = self._act_fn(op, self.w1(op, hidden_states))
        up = self.w3(op, hidden_states)
        return self.w2(op, op.Mul(gate, up))


class _InternLMDecoderLayer(nn.Module):
    """Pre-norm decoder layer matching HF InternLM naming.

    Uses attention_norm/ffn_norm instead of input_layernorm/post_attention_layernorm,
    and attention/feed_forward instead of self_attn/mlp.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = _InternLMMLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(op, hidden_states)
        hidden_states, present_kv = self.attention(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _InternLMTextModel(nn.Module):
    """InternLM text model backbone.

    Uses standard layers/norm naming (matches HF) but custom decoder
    layers with HF-aligned attribute names.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_InternLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
    ):
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


class InternLM2CausalLMModel(nn.Module):
    """InternLM2 causal language model with HF-aligned weight names.

    Module attributes match HF naming to minimize preprocess_weights.
    Only tok_embeddings→embed_tokens, wo→o_proj renames and the
    grouped/interleaved wqkv split remain.

    The wqkv weight has a grouped/interleaved layout:
    [num_kv_heads, (num_q_per_kv_group + 2), head_dim, hidden_size]
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = _InternLMTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
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
        """Map HF InternLM2 weights to our parameter names.

        After attribute alignment, only three operations remain:
        1. tok_embeddings → embed_tokens (HF uses tok_embeddings)
        2. wo → o_proj (Attention component uses o_proj)
        3. Grouped/interleaved wqkv split into separate q/k/v
        """
        # Weight tying
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        num_kv_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        group_size = num_kv_groups + 2

        for key in list(state_dict.keys()):
            # Rename: tok_embeddings → embed_tokens
            if ".tok_embeddings." in key:
                new_key = key.replace(".tok_embeddings.", ".embed_tokens.")
                state_dict[new_key] = state_dict.pop(key)
                key = new_key

            # Rename: wo → o_proj (Attention component uses o_proj)
            if ".wo." in key:
                new_key = key.replace(".wo.", ".o_proj.")
                state_dict[new_key] = state_dict.pop(key)
                key = new_key

            # Split grouped/interleaved wqkv weight
            if ".attention.wqkv.weight" in key:
                weight = state_dict.pop(key)
                wqkv_grouped = weight.reshape(
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                    self.config.hidden_size,
                )
                q_weight = wqkv_grouped[:, :num_kv_groups, :, :].reshape(
                    q_size, self.config.hidden_size
                )
                k_weight = wqkv_grouped[:, -2, :, :].reshape(kv_size, self.config.hidden_size)
                v_weight = wqkv_grouped[:, -1, :, :].reshape(kv_size, self.config.hidden_size)
                prefix = key.replace("wqkv.weight", "")
                state_dict[f"{prefix}q_proj.weight"] = q_weight
                state_dict[f"{prefix}k_proj.weight"] = k_weight
                state_dict[f"{prefix}v_proj.weight"] = v_weight

            elif ".attention.wqkv.bias" in key:
                bias = state_dict.pop(key)
                bias_grouped = bias.reshape(
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                )
                q_bias = bias_grouped[:, :num_kv_groups, :].reshape(q_size)
                k_bias = bias_grouped[:, -2, :].reshape(kv_size)
                v_bias = bias_grouped[:, -1, :].reshape(kv_size)
                prefix = key.replace("wqkv.bias", "")
                state_dict[f"{prefix}q_proj.bias"] = q_bias
                state_dict[f"{prefix}k_proj.bias"] = k_bias
                state_dict[f"{prefix}v_proj.bias"] = v_bias

        return state_dict
