# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multimodal model task (vision + audio + text)."""

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


class MultiModalTask(ModelTask):
    """Multimodal model that processes images, audio, and text.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, total_seq_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - pixel_values: [batch, channels, height, width] FLOAT
        - audio_features: [batch, audio_seq_len, num_mel_bins] FLOAT
        - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim] FLOAT
        - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim] FLOAT

    Outputs:
        - logits: FLOAT
        - present.{i}.key: FLOAT
        - present.{i}.value: FLOAT

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, pixel_values, audio_features, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        image_size = config.vision.image_size or 224 if config.vision else 224
        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, 3, image_size, image_size]),
            type=ir.TensorType(config.dtype),
        )

        audio_input_size = (config.audio.input_size if config.audio else None) or 80
        audio_features = ir.Value(
            name="audio_features",
            shape=ir.Shape([batch, "audio_seq_len", audio_input_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [input_ids, attention_mask, position_ids, pixel_values, audio_features]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            audio_features=audio_features,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values, past_key_values=past_key_values)

        return ModelPackage({"model": _make_model(graph)}, config=config)
