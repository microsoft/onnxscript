# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio feature extraction task.

Builds a single ONNX graph for encoder-only audio models (Wav2Vec2, HuBERT, etc.)
that take raw waveform input and produce hidden state outputs.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class AudioFeatureExtractionTask(ModelTask):
    """Build ONNX graph for audio feature extraction (encoder-only)."""

    name = "audio-feature-extraction"

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        input_values = ir.Value(
            name="input_values",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", "time")),
        )

        graph, builder = _make_graph([input_values])
        op = builder.op

        last_hidden_state = module(op, input_values=input_values)

        last_hidden_state.name = "last_hidden_state"
        graph.outputs.append(last_hidden_state)

        return ModelPackage({"model": _make_model(graph)})


class ClapAudioFeatureExtractionTask(ModelTask):
    """Build ONNX graph for CLAP audio feature extraction.

    Accepts mel-spectrogram input ``(batch, time, freq)`` rather than
    raw waveform, matching ``ClapAudioModel.forward(input_features=...)``.
    """

    name = "clap-audio-feature-extraction"

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        assert config.audio is not None, "ClapAudioFeatureExtractionTask requires AudioConfig"

        input_features = ir.Value(
            name="input_features",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", "time", "freq")),
        )

        graph, builder = _make_graph([input_features])
        op = builder.op

        audio_embeds = module(op, input_features=input_features)

        audio_embeds.name = "audio_embeds"
        graph.outputs.append(audio_embeds)

        return ModelPackage({"model": _make_model(graph)})
