# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR: Audio speech recognition with Qwen3 text decoder.

Architecture:
  - Audio encoder: 3x Conv2d downsampling → sinusoidal PE → N bidirectional
    encoder layers → LayerNorm → proj1 → GELU → proj2
  - Text decoder: Qwen3 with QK norm + interleaved MRoPE
  - Fusion: Audio features replace audio_token_id positions in text embeddings

Also supports Qwen3-ForcedAligner (same architecture, classification head).

Reference: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
HuggingFace class: Qwen3ASRForConditionalGeneration
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import (
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
)
from mobius.components._conv import Conv2d
from mobius.components._decoder import DecoderLayer
from mobius.components._qwen3_asr_audio import (
    Qwen3ASRAudioEncoderLayer,
)
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import initialize_rope


def _sinusoidal_position_embedding(max_positions: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional embeddings matching Qwen3-ASR.

    Uses log-timescale increments (different from Whisper which uses
    alternating sin/cos layout). Layout: [sin_0..sin_n, cos_0..cos_n].
    """
    channels = d_model
    log_timescale_increment = np.log(10000.0) / (channels // 2 - 1)
    inv_timescales = np.exp(
        -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
    )
    scaled_time = (
        np.arange(max_positions, dtype=np.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )
    # Layout: [sin, cos] matching HF SinusoidsPositionEmbedding
    pe = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)
    return pe


class Qwen3ASRAudioEncoder(nn.Module):
    """Qwen3-ASR audio encoder.

    Converts mel spectrogram to audio feature embeddings:
      mel (batch, num_mel_bins, seq_len)
      → 3x Conv2d with GELU downsampling
      → linear projection (conv_out)
      → sinusoidal positional embeddings
      → N bidirectional encoder layers
      → LayerNorm (ln_post)
      → proj1 → GELU → proj2

    Output: (batch, out_seq_len, output_dim)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        d_model = audio.d_model or 896
        num_mel_bins = audio.num_mel_bins or 128
        encoder_layers = audio.encoder_layers or 18
        encoder_heads = audio.encoder_attention_heads or 14
        encoder_ffn = audio.encoder_ffn_dim or 3584
        max_source_positions = audio.max_source_positions or 1500
        downsample_hidden_size = audio.downsample_hidden_size or 480
        output_dim = audio.output_dim or 1024

        # 3x Conv2d downsampling: (1, mel, seq) → (dhs, mel//8, seq//8)
        self.conv2d1 = Conv2d(
            1,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d2 = Conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = Conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Linear projection from flattened conv features to d_model
        # After 3 strides of 2: freq_dim = (((mel+1)//2+1)//2+1)//2
        freq_after_conv = (((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        conv_out_dim = downsample_hidden_size * freq_after_conv
        self.conv_out = Linear(conv_out_dim, d_model, bias=False)

        # Sinusoidal positional embeddings (frozen)
        pe_data = _sinusoidal_position_embedding(max_source_positions, d_model)
        self.positional_embedding = nn.Parameter(
            [max_source_positions, d_model],
            name="positional_embedding.positional_embedding",
            data=ir.tensor(pe_data),
        )

        # Encoder transformer layers
        self.layers = nn.ModuleList(
            [
                Qwen3ASRAudioEncoderLayer(d_model, encoder_heads, encoder_ffn)
                for _ in range(encoder_layers)
            ]
        )

        # Post-encoder normalization
        self.ln_post = LayerNorm(d_model)

        # Output projection: d_model → output_dim
        self.proj1 = Linear(d_model, d_model, bias=True)
        self.proj2 = Linear(d_model, output_dim, bias=True)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        """Encode mel spectrogram to audio features.

        Args:
            input_features: (batch, num_mel_bins, seq_len) mel spectrogram

        Returns:
            audio_features: (batch, out_seq_len, output_dim)
        """
        # Add channel dimension: (batch, mel, seq) → (batch, 1, mel, seq)
        hidden_states = op.Unsqueeze(input_features, [1])

        # 3x Conv2d downsampling with GELU
        hidden_states = op.Gelu(self.conv2d1(op, hidden_states))
        hidden_states = op.Gelu(self.conv2d2(op, hidden_states))
        hidden_states = op.Gelu(self.conv2d3(op, hidden_states))

        # Reshape: (batch, channels, freq, time) → (batch, time, channels*freq)
        # Permute to (batch, time, channels, freq) then flatten last two dims
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        batch_dim = op.Shape(hidden_states, start=0, end=1)
        time_dim = op.Shape(hidden_states, start=1, end=2)
        new_shape = op.Concat(batch_dim, time_dim, op.Constant(value_ints=[-1]), axis=0)
        hidden_states = op.Reshape(hidden_states, new_shape)

        # Linear projection to d_model
        hidden_states = self.conv_out(op, hidden_states)

        # Add sinusoidal positional embeddings (up to seq_len)
        seq_len = op.Shape(hidden_states, start=1, end=2)
        # Slice PE to match actual sequence length
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0]),
        )
        hidden_states = op.Add(hidden_states, pe_slice)

        # Encoder layers
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        # Post-encoder norm + projection
        hidden_states = self.ln_post(op, hidden_states)
        hidden_states = self.proj1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.proj2(op, hidden_states)

        return hidden_states


class Qwen3ASREmbeddingModel(nn.Module):
    """Qwen3-ASR embedding model: fuses text and audio embeddings.

    Replaces audio_token_id positions in the text embedding with
    audio features from the audio encoder.

    Inputs:
        input_ids: (batch, seq_len) token IDs
        audio_features: (num_audio_tokens, output_dim) from audio encoder

    Output:
        inputs_embeds: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        audio_token_id = config.audio.audio_token_id if config.audio else 151676
        self._audio_token_id = audio_token_id

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ):
        """Fuse text embeddings with audio features.

        Audio features replace positions where input_ids == audio_token_id.
        Uses Gather + Where pattern (equivalent to masked_scatter).
        """
        # Text embeddings: (batch, seq_len, hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Create mask: True where input_ids == audio_token_id
        audio_token = op.Constant(value_int=self._audio_token_id)
        is_audio = op.Equal(input_ids, audio_token)
        # Unsqueeze for broadcasting: (batch, seq_len, 1)
        is_audio_3d = op.Unsqueeze(is_audio, [-1])

        # Pad audio_features with a zero row at index 0 to handle
        # the case where no audio tokens exist in the sequence
        feature_dim = op.Shape(audio_features, start=1, end=2)
        zero_row_shape = op.Concat(op.Constant(value_ints=[1]), feature_dim, axis=0)
        zero_row = op.ConstantOfShape(
            zero_row_shape,
            value=ir.tensor(np.zeros(1, dtype=np.float32)),
        )
        # Prepend zero row: (num_audio_tokens + 1, output_dim)
        padded_features = op.Concat(zero_row, audio_features, axis=0)

        # Compute gather indices: cumulative sum of audio mask gives
        # 1-based indices into padded_features (0 = zero padding row)
        is_audio_int = op.Cast(is_audio, to=7)  # INT64
        # Flatten across batch for cumsum then reshape
        flat_mask = op.Reshape(is_audio_int, op.Constant(value_ints=[-1]))
        flat_indices = op.CumSum(flat_mask, op.Constant(value_int=0))
        flat_indices = op.Mul(flat_indices, flat_mask)
        # Reshape back to (batch, seq_len)
        indices = op.Reshape(flat_indices, op.Shape(input_ids))

        # Gather audio features using computed indices
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where: replace audio positions with gathered features
        inputs_embeds = op.Where(is_audio_3d, gathered, inputs_embeds)

        return inputs_embeds


class Qwen3ASRDecoderModel(nn.Module):
    """Qwen3-ASR text decoder: inputs_embeds → logits + KV cache.

    Standard Qwen3 decoder with QK norm and interleaved MRoPE.
    Takes inputs_embeds (fused text+audio) instead of input_ids.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
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
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Qwen3-ASR composite model for speech recognition.

    Contains:
    - ``audio_tower``: Audio encoder (mel → audio features)
    - ``embedding``: Text+audio embedding fusion
    - ``decoder``: Text decoder with KV cache

    Also supports Qwen3-ForcedAligner when ``classify_num`` is set
    in the audio config (uses classification head instead of LM head).

    HuggingFace class: ``Qwen3ASRForConditionalGeneration``
    """

    default_task: str = "speech-language"
    category: str = "Speech-to-Text"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # Determine output vocab: classify_num for ForcedAligner,
        # else vocab_size
        output_vocab = config.vocab_size
        if config.audio and config.audio.classify_num:
            output_vocab = config.audio.classify_num

        self.audio_tower = Qwen3ASRAudioEncoder(config)

        self.embedding = Qwen3ASREmbeddingModel(config)
        self.decoder = Qwen3ASRDecoderModel(
            config
            if output_vocab == config.vocab_size
            else dataclasses.replace(config, vocab_size=output_vocab)
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for text-generation task.

        Embeds input_ids using the text embedding (no audio fusion in
        this path — audio features are fused externally), then runs
        the decoder to produce logits and KV cache.
        """
        inputs_embeds = self.embedding.embed_tokens(op, input_ids)
        return self.decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to ONNX module structure.

        HF weights have ``thinker.`` prefix:
        - ``thinker.audio_tower.*`` → ``audio_tower.*``
        - ``thinker.model.*`` → text decoder layers
        - ``thinker.lm_head.*`` → ``decoder.lm_head.*``

        For embedding:
            ``thinker.model.embed_tokens.weight``
            → ``embedding.embed_tokens.weight``
        For decoder layers:
            ``thinker.model.layers.N.*`` → ``decoder.layers.N.*``
        For decoder norm:
            ``thinker.model.norm.*`` → ``decoder.norm.*``
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Strip thinker. prefix
            if key.startswith("thinker."):
                key = key[len("thinker.") :]

            # Route audio_tower weights
            if key.startswith("audio_tower."):
                cleaned[key] = value
                continue

            # Route lm_head to decoder
            if key.startswith("lm_head."):
                cleaned[f"decoder.{key}"] = value
                continue

            # Route model.* to appropriate sub-module
            if key.startswith("model."):
                inner = key[len("model.") :]

                # embed_tokens → embedding module
                if inner.startswith("embed_tokens."):
                    cleaned[f"embedding.{inner}"] = value
                    continue

                # layers.N.* and norm.* → decoder module
                if inner.startswith(("layers.", "norm.")):
                    cleaned[f"decoder.{inner}"] = value
                    continue

                # rotary_emb → decoder module
                if inner.startswith("rotary_emb."):
                    cleaned[f"decoder.{inner}"] = value
                    continue

            cleaned[key] = value

        # Weight tying: embed_tokens → lm_head
        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
