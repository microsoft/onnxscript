# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mllama (Meta Llama 3.2 Vision) multimodal model.

Mllama uses a fundamentally different multimodal approach than LLaVA:
- LLaVA concatenates vision tokens into the text sequence
- Mllama uses **interleaved cross-attention layers** in the text decoder

The text decoder has two types of layers:
1. **Self-attention layers** (standard Llama-style) — most layers
2. **Cross-attention layers** (at specific layer indices) — attend to vision features
   with tanh-gated residual connections

The vision encoder is a ViT-based model that produces image features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, MllamaConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    MLP,
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    VisionModel,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class MllamaCrossAttention(nn.Module):
    """Cross-attention module for Mllama.

    Queries come from the text decoder hidden states.
    Keys and values come from the vision encoder output.
    Uses QK-norm (RMSNorm on Q and K projections).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = float(self.head_dim**-0.5)

        self.q_proj = Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cross_attention_states: ir.Value,
        past_key_value: tuple | None = None,
    ):
        """Cross-attention forward.

        Args:
            hidden_states: Text decoder hidden states (queries).
            cross_attention_states: Vision encoder output (keys/values).
            past_key_value: Cached (key, value) for incremental decoding.

        Returns:
            Tuple of (output, (present_key, present_value)).
        """
        query_states = self.q_proj(op, hidden_states)
        # Cross-attention K/V caching: on prefill cross_attention_states has
        # the full vision tokens; on decode the runtime passes a 0-length
        # tensor so the projection is essentially free.  The op.Attention
        # concat of 0-length new K/V with past yields past unchanged,
        # so the cross-attention cache stays constant after prefill.
        key_states = self.k_proj(op, cross_attention_states)
        value_states = self.v_proj(op, cross_attention_states)

        # Apply QK-norm per head
        query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
        key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        if past_key_value is not None:
            past_key, past_value = past_key_value
        else:
            past_key = None
            past_value = None

        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            None,  # No attention bias for cross-attention
            past_key,
            past_value,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            is_causal=0,
            _outputs=3,
        )

        output = self.o_proj(op, attn_output)
        return output, (present_key, present_value)


class MllamaCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention decoder layer with tanh-gated residual connections.

    The cross-attention and MLP outputs are gated by learnable scalar parameters
    (initialized to zero) passed through tanh, enabling gradual integration of
    vision features during training.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.cross_attn = MllamaCrossAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Tanh gates initialized to zero
        self.cross_attn_attn_gate = nn.Parameter([1])
        self.cross_attn_mlp_gate = nn.Parameter([1])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cross_attention_states: ir.Value,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_kv = self.cross_attn(
            op,
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
        )

        # Tanh-gated residual for attention
        gate = op.Tanh(self.cross_attn_attn_gate)
        gated_attn = op.Mul(attn_output, gate)
        hidden_states = op.Add(residual, gated_attn)

        # MLP with tanh-gated residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        mlp_output = self.mlp(op, hidden_states)
        mlp_gate = op.Tanh(self.cross_attn_mlp_gate)
        gated_mlp = op.Mul(mlp_output, mlp_gate)
        hidden_states = op.Add(residual, gated_mlp)

        return hidden_states, present_kv


class MllamaTextModel(nn.Module):
    """Mllama text decoder with interleaved self-attention and cross-attention layers.

    Self-attention layers use standard Llama-style causal attention with RoPE.
    Cross-attention layers attend to vision encoder features at specific layer indices.
    """

    def __init__(self, config: MllamaConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

        # Build interleaved layers
        cross_attention_layers = set(config.cross_attention_layers or [])
        self.cross_attention_layers = cross_attention_layers
        self.layers = nn.ModuleList()
        self.layer_is_cross_attention = []

        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in cross_attention_layers:
                self.layers.append(MllamaCrossAttentionDecoderLayer(config))
                self.layer_is_cross_attention.append(True)
            else:
                self.layers.append(DecoderLayer(config))
                self.layer_is_cross_attention.append(False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        cross_attention_states: ir.Value | None = None,
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
            input_ids=input_ids if input_ids is not None else hidden_states,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)

        for layer, is_cross, past_kv in zip(
            self.layers, self.layer_is_cross_attention, past_kvs
        ):
            if is_cross:
                hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states,
                    past_key_value=past_kv,
                )
            else:
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


class _MllamaDecoderModel(nn.Module):
    """Mllama text decoder sub-model taking inputs_embeds + cross_attention_states."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = MllamaTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        cross_attention_states: ir.Value | None = None,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _MllamaVisionEncoderModel(nn.Module):
    """Mllama vision encoder: ViT-based."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_model = VisionModel(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        return self.vision_model(op, pixel_values)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value for key, value in state_dict.items() if key.startswith("vision_model.")
        }


class _MllamaEmbeddingModel(nn.Module):
    """Mllama token embedding model."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text_embeds = self.embed_tokens(op, input_ids)
        # For Mllama, vision features are injected via cross-attention
        # (not concatenated). The embedding model just returns text embeds.
        # image_features are passed separately to the decoder.
        return text_embeds

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


class MllamaCausalLMModel(nn.Module):
    """Mllama vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: text decoder with cross-attention for vision features
    - vision_encoder: ViT-based vision encoder
    - embedding: token embedding
    """

    default_task: str = "mllama-vision-language"
    category: str = "Multimodal"
    config_class: type = MllamaConfig

    def __init__(self, config: MllamaConfig):
        super().__init__()
        self.config = config
        self.decoder = _MllamaDecoderModel(config)
        self.vision_encoder = _MllamaVisionEncoderModel(config)
        self.embedding = _MllamaEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "MllamaCausalLMModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict
