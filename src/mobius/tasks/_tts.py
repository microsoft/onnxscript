# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""TTS 4-model split task for Qwen3-TTS.

Builds four separate ONNX models:
1. **talker**: inputs_embeds → logits (first code group) + last_hidden_state + KV cache
2. **code_predictor**: inputs_embeds → hidden_states + KV cache (1D RoPE)
3. **embedding**: text_ids + codec_ids → text_embeds + codec_embeds
4. **speaker_encoder**: mel_input → speaker_embedding

Used by Qwen3TTSForConditionalGeneration.
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


class TTSTask(ModelTask):
    """4-model split for Qwen3-TTS.

    The module must provide four sub-modules as attributes:

    - ``talker``: Decoder producing logits + last_hidden_state
    - ``code_predictor``: Small decoder for remaining code groups
    - ``embedding``: Text + codec embedding model
    - ``speaker_encoder``: ECAPA-TDNN speaker encoder

    Each sub-module is wired into its own ONNX graph.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["talker"] = self._build_talker(module.talker, config)
        models["code_predictor"] = self._build_code_predictor(module.code_predictor, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        if module.speaker_encoder is not None:
            models["speaker_encoder"] = self._build_speaker_encoder(
                module.speaker_encoder, config
            )

        return ModelPackage(models, config=config)

    def _build_talker(
        self,
        talker: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build talker: inputs_embeds → logits + last_hidden_state + KV cache.

        Uses MRoPE 3D position_ids (3, batch, seq_len).
        """
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

        graph, builder = _make_graph(graph_inputs, name="talker")
        logits, last_hidden_state, present_key_values = talker(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        last_hidden_state.name = "last_hidden_state"
        graph.outputs.append(logits)
        graph.outputs.append(last_hidden_state)
        _register_kv_cache_outputs(graph, present_key_values, past_key_values=past_key_values)
        return _make_model(graph)

    def _build_code_predictor(
        self,
        code_predictor: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build code predictor: inputs_embeds → logits + KV cache.

        The generation loop constructs inputs_embeds in **talker_hidden**
        space (e.g. 2048 for 1.7B, 1024 for 0.6B):
          - Step 0 (prefill): concat(talker_hidden, talker_embed(code_0))
            → 2 tokens. Matches HF's code predictor prefill.
          - Steps 1-14: CP_embed[step-1](code_i) → 1 token.
            CP codec embeddings are stored in talker_hidden space.

        The model projects to cp_hidden internally via
        ``small_to_mtp_projection`` (Identity when dims match).

        Uses standard 1D RoPE (2D position_ids).
        """
        # Read code predictor config directly to avoid importing the model class.
        # Defaults match Qwen3TTSCodePredictorModel._make_cp_config().
        tts = config.tts
        cp = tts.code_predictor if tts else None
        cp_num_hidden_layers = cp.num_hidden_layers if cp else 5
        cp_num_key_value_heads = cp.num_key_value_heads if cp else 8
        cp_head_dim = cp.head_dim if cp else 128

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        # Pre-embedded input in talker_hidden space (constructed by
        # generation loop). The model projects to cp_hidden internally.
        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        # Step index: selects which lm_head to use (0..14)
        step_index = ir.Value(
            name="step_index",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # 1D RoPE: 2D position_ids (batch, seq_len)
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [
            inputs_embeds,
            step_index,
            attention_mask,
            position_ids,
        ]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            cp_num_hidden_layers,
            cp_num_key_value_heads,
            cp_head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="code_predictor")
        logits, present_key_values, codec_embeddings = code_predictor(
            builder.op,
            inputs_embeds=inputs_embeds,
            step_index=step_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        # Expose stacked codec embeddings for generation loop to extract.
        # The Identity node ensures renaming the output doesn't affect
        # the initializer name used for weight loading.
        codec_embeddings.name = "codec_embeddings"
        graph.outputs.append(codec_embeddings)
        _register_kv_cache_outputs(graph, present_key_values, past_key_values=past_key_values)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: text_ids + codec_ids → text_embeds + codec_embeds."""
        batch = ir.SymbolicDim("batch")
        text_seq = ir.SymbolicDim("text_sequence_len")
        codec_seq = ir.SymbolicDim("codec_sequence_len")

        text_ids = ir.Value(
            name="text_ids",
            shape=ir.Shape([batch, text_seq]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        codec_ids = ir.Value(
            name="codec_ids",
            shape=ir.Shape([batch, codec_seq]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([text_ids, codec_ids], name="embedding")
        text_embeds, codec_embeds = embedding(
            builder.op,
            text_ids=text_ids,
            codec_ids=codec_ids,
        )

        text_embeds.name = "text_embeds"
        codec_embeds.name = "codec_embeds"
        graph.outputs.append(text_embeds)
        graph.outputs.append(codec_embeds)
        return _make_model(graph)

    def _build_speaker_encoder(
        self,
        speaker_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build speaker encoder: mel_input → speaker_embedding."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        tts = config.tts
        se = tts.speaker_encoder if tts else None
        mel_dim = se.mel_dim if se else 128

        mel_input = ir.Value(
            name="mel_input",
            shape=ir.Shape([batch, mel_seq, mel_dim]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([mel_input], name="speaker_encoder")
        speaker_embedding = speaker_encoder(builder.op, mel_input)

        speaker_embedding.name = "speaker_embedding"
        graph.outputs.append(speaker_embedding)
        return _make_model(graph)
