# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Whisper encoder-decoder model for speech-to-text.

Implements the Whisper architecture as two separate traceable modules:
- ``WhisperEncoderModel``: mel features → encoder hidden states
- ``WhisperDecoderModel``: decoder input IDs + encoder output → logits + KV cache

``WhisperForConditionalGeneration`` holds both and provides ``preprocess_weights()``.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import WhisperConfig
from mobius.components import (
    Embedding,
    LayerNorm,
    Linear,
)
from mobius.components._whisper import (
    Conv1d,
    WhisperDecoderLayer,
    WhisperEncoderLayer,
)


def _sinusoidal_positional_embedding(max_positions: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional embeddings (frozen, not learned).

    Returns a [max_positions, d_model] float32 array matching
    ``WhisperPositionalEmbedding`` in HuggingFace transformers.
    """
    position = np.arange(max_positions, dtype=np.float32)[:, np.newaxis]
    half_dim = d_model // 2
    div_term = np.exp(np.arange(half_dim, dtype=np.float32) * -(math.log(10000.0) / half_dim))
    pe = np.zeros((max_positions, d_model), dtype=np.float32)
    pe[:, :half_dim] = np.sin(position * div_term)
    pe[:, half_dim:] = np.cos(position * div_term)
    return pe


class WhisperEncoderModel(nn.Module):
    """Whisper encoder: mel features → encoder hidden states.

    Architecture: Conv1d x 2 → sinusoidal positional embeddings → encoder layers → LayerNorm.

    Input:  ``[batch, num_mel_bins, audio_seq_len]``
    Output: ``[batch, audio_seq_len // 2, d_model]``
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        d_model = config.hidden_size
        num_mel_bins = config.num_mel_bins
        encoder_layers = config.encoder_layers or config.num_hidden_layers
        encoder_heads = config.encoder_attention_heads or config.num_attention_heads
        encoder_ffn = config.encoder_ffn_dim or config.intermediate_size
        max_source_positions = config.max_source_positions or 1500
        activation = config.hidden_act or "gelu"

        eps = config.layer_norm_eps

        self.conv1 = Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Sinusoidal positional embeddings (frozen)
        pe_data = _sinusoidal_positional_embedding(max_source_positions, d_model)
        self.embed_positions = nn.Parameter(
            [max_source_positions, d_model],
            name="embed_positions.weight",
            data=ir.tensor(pe_data),
        )

        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(d_model, encoder_heads, encoder_ffn, activation, eps=eps)
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm = LayerNorm(d_model, eps=eps)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        # input_features: [batch, num_mel_bins, audio_seq_len]
        hidden_states = op.Gelu(self.conv1(op, input_features))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))

        # Transpose: [batch, d_model, seq_len//2] → [batch, seq_len//2, d_model]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Add sinusoidal positional embeddings
        hidden_states = op.Add(hidden_states, self.embed_positions)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.layer_norm(op, hidden_states)
        return hidden_states


class WhisperDecoderModel(nn.Module):
    """Whisper decoder: token IDs + encoder output → logits + KV cache.

    Architecture: token embed + positional embed → decoder layers → LayerNorm → proj_out.

    The decoder uses causal self-attention with KV caching and
    cross-attention to encoder hidden states.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        d_model = config.hidden_size
        decoder_heads = config.num_attention_heads
        decoder_ffn = config.intermediate_size
        decoder_layers = config.num_hidden_layers
        max_target_positions = config.max_target_positions
        activation = config.hidden_act or "gelu"

        eps = config.layer_norm_eps

        self.embed_tokens = Embedding(config.vocab_size, d_model, config.pad_token_id)
        self.embed_positions = Embedding(max_target_positions, d_model)

        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(d_model, decoder_heads, decoder_ffn, activation, eps=eps)
                for _ in range(decoder_layers)
            ]
        )
        self.layer_norm = LayerNorm(d_model, eps=eps)
        self.proj_out = Linear(d_model, config.vocab_size, bias=False)

        self._scale_embedding = config.scale_embedding
        self._embed_scale = math.sqrt(d_model) if config.scale_embedding else 1.0

    def forward(
        self,
        op: builder.OpBuilder,
        decoder_input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # Token embeddings + positional embeddings
        hidden_states = self.embed_tokens(op, decoder_input_ids)
        if self._scale_embedding:
            hidden_states = op.Mul(hidden_states, self._embed_scale)
        positions = self.embed_positions(op, position_ids)
        hidden_states = op.Add(hidden_states, positions)

        # Decoder layers (causal masking handled by is_causal attr in Attention op)
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.layer_norm(op, hidden_states)
        logits = self.proj_out(op, hidden_states)
        return logits, present_key_values


class WhisperForConditionalGeneration(nn.Module):
    """Whisper encoder-decoder model for speech recognition and transcription.

    This class holds both ``WhisperEncoderModel`` and ``WhisperDecoderModel``
    as sub-modules. Use ``SpeechToTextTask.build()`` to trace them
    into separate ONNX models.
    """

    default_task: str = "speech-to-text"
    category: str = "Speech-to-Text"
    config_class: type = WhisperConfig

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.model = _WhisperModel(config)
        self.proj_out = self.model.decoder.proj_out

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF weight names to our module structure.

        HF weights are named ``model.encoder.X`` / ``model.decoder.X``.
        Our ONNX graphs use ``encoder.X`` / ``decoder.X`` (no ``model.`` prefix)
        because ``SpeechToTextTask`` traces ``module.model.encoder`` directly.
        """
        # Weight tying: proj_out shares weights with decoder embed_tokens
        if self.config.tie_word_embeddings:
            embed_key = "model.decoder.embed_tokens.weight"
            proj_key = "proj_out.weight"
            if proj_key in state_dict and embed_key not in state_dict:
                state_dict[embed_key] = state_dict[proj_key]
            elif embed_key in state_dict and proj_key not in state_dict:
                state_dict[proj_key] = state_dict[embed_key]
            # Remap proj_out to be under model.decoder
            if proj_key in state_dict:
                state_dict[f"model.decoder.{proj_key}"] = state_dict.pop(proj_key)

        # Strip "model." prefix: model.encoder.X → encoder.X, model.decoder.X → decoder.X
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                remapped[key[len("model.") :]] = value
            else:
                remapped[key] = value
        return remapped


class _WhisperModel(nn.Module):
    """Inner model holding encoder + decoder (matches HF model.encoder/model.decoder naming)."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.encoder = WhisperEncoderModel(config)
        self.decoder = WhisperDecoderModel(config)
