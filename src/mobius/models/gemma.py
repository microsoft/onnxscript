# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma and Gemma2 causal language models.

Replicates HuggingFace's ``GemmaForCausalLM`` and ``Gemma2ForCausalLM``.
Key differences from the base CausalLMModel:
- RMSNorm uses +1 offset (``OffsetRMSNorm``) and embedding scaling.
- Gemma2 adds attention logit soft-capping, final logit soft-capping,
  and alternating local/global sliding-window attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, Gemma2Config
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    OffsetRMSNorm,
    create_attention_bias,
    create_decoder_layer,
    initialize_rope,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class GemmaScaledWordEmbedding(Embedding):
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


class GemmaTextModel(nn.Module):
    """Gemma text model with embedding scaling and +1 norm offset."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        embed_scale = float(np.round(np.sqrt(config.hidden_size), decimals=2))
        self.embed_tokens = GemmaScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [
                create_decoder_layer(config, norm_class=OffsetRMSNorm)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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


class GemmaCausalLMModel(CausalLMModel):
    """Gemma causal language model with +1 RMSNorm offset and embedding scaling.

    Replicates HuggingFace's ``GemmaForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = GemmaTextModel(config)


class Gemma2Attention(Attention):
    """Gemma2 attention with soft-capping on attention logits.

    Applies tanh soft-capping to QK^T scores before softmax:
        scores = tanh(scores / softcap) * softcap

    This bounds attention weights to [-softcap, softcap], preventing
    extreme attention concentrations. Uses the ONNX Attention op's
    native ``softcap`` attribute (opset 23).

    Also uses ``query_pre_attn_scalar`` for attention scaling instead
    of the default ``1/sqrt(head_dim)``.

    Replicates HuggingFace's ``Gemma2Attention``.
    """

    def __init__(self, config: Gemma2Config):
        # Use query_pre_attn_scalar for attention scaling if available
        scale = None
        if config.query_pre_attn_scalar:
            scale = config.query_pre_attn_scalar**-0.5
        super().__init__(config, rms_norm_class=OffsetRMSNorm, scale=scale)
        self._softcap = config.attn_logit_softcapping

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
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

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

        # ONNX Attention op (opset 23) with native softcap support
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
            softcap=self._softcap,
            _outputs=3,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class Gemma2DecoderLayer(nn.Module):
    """Gemma2 decoder layer with extra pre/post feedforward layer norms."""

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.self_attn = Gemma2Attention(config)
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


class Gemma2TextModel(nn.Module):
    """Gemma2 text model with alternating local/global attention."""

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self._dtype = config.dtype
        embed_scale = float(np.round(np.sqrt(config.hidden_size), decimals=2))
        self.embed_tokens = GemmaScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.sliding_window = config.sliding_window
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def _is_local(self, layer_id: int) -> bool:
        """Gemma2 uses alternating attention: even=global, odd=local."""
        return layer_id % 2 == 1

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
            attn_bias = sliding_attn_bias if self._is_local(i) else full_attn_bias
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


class Gemma2CausalLMModel(CausalLMModel):
    """Gemma2 causal language model with alternating local/global attention.

    Applies final logit soft-capping after LM head projection:
        logits = tanh(logits / softcap) * softcap

    Replicates HuggingFace's ``Gemma2ForCausalLM``.
    """

    config_class: type = Gemma2Config

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.model = Gemma2TextModel(config)
        self._final_logit_softcapping = config.final_logit_softcapping

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

        # Final logit soft-capping: tanh(logits/cap) * cap
        if self._final_logit_softcapping > 0.0:
            logits = op.Div(logits, self._final_logit_softcapping)
            logits = op.Tanh(logits)
            logits = op.Mul(logits, self._final_logit_softcapping)

        return logits, present_key_values
