# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Speech-language 3-model split task for ASR / forced alignment.

Builds three separate ONNX models:
1. **audio_encoder**: input_features (mel spectrogram) → audio_features
2. **embedding**: input_ids + audio_features → inputs_embeds
3. **decoder**: inputs_embeds → logits + KV cache (MRoPE 3D position_ids)

Used by Qwen3-ASR and Qwen3-ForcedAligner.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
    _register_kv_cache_outputs,
)


class SpeechLanguageTask(ModelTask):
    """3-model split for speech-language models (ASR / forced alignment).

    The module must provide three sub-modules as attributes:

    - ``audio_tower``: audio encoder taking ``input_features`` (mel)
    - ``embedding``: embedding model fusing text + audio features
    - ``decoder``: text decoder taking ``inputs_embeds`` with KV cache

    Each sub-module is wired into its own ONNX graph.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["audio_encoder"] = self._build_audio_encoder(module.audio_tower, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = self._build_decoder(module.decoder, config)

        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: mel (batch, n_mels, time) → audio features."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        n_mels = config.audio.num_mel_bins or 128 if config.audio else 128

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, n_mels, mel_seq]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_features], name="audio_encoder")
        audio_features = audio_encoder(builder.op, input_features)

        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: input_ids + audio_features → inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
        output_dim = (
            config.audio.output_dim or config.hidden_size
            if config.audio
            else config.hidden_size
        )

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        audio_features = ir.Value(
            name="audio_features",
            shape=ir.Shape([num_audio_tokens, output_dim]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_ids, audio_features], name="embedding")
        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
            audio_features=audio_features,
        )

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)
        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder with MRoPE 3D position_ids."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # MRoPE: 3D position_ids (3, batch, seq_len)
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([3, batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        logits, present_key_values = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)
        return _make_model(graph)
