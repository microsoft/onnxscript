# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Codec tokenizer 2-model task for Qwen3-TTS-Tokenizer-12Hz.

Builds two ONNX models:
1. **decoder**: codes (B, 16, T) → waveform (B, 1, T*1920)
2. **encoder**: waveform (B, 1, samples) → codes (B, 16, T)
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


class CodecTask(ModelTask):
    """2-model split for Qwen3-TTS codec tokenizer.

    The module must provide two sub-modules:
    - ``decoder``: codes → waveform
    - ``encoder``: waveform → codes

    Each is wired into its own ONNX graph with no KV cache.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["encoder"] = self._build_encoder(module.encoder, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder: codes → waveform.

        Inputs:
            codes: (B, num_quantizers, T) int64
        Outputs:
            waveform: (B, 1, T * upsample_factor) float32
        """
        batch = ir.SymbolicDim("batch")
        num_q = ir.SymbolicDim("num_quantizers")
        seq_len = ir.SymbolicDim("sequence_len")

        codes = ir.Value(
            name="codes",
            shape=ir.Shape([batch, num_q, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([codes], name="decoder")
        waveform = decoder(builder.op, codes)

        waveform.name = "waveform"
        graph.outputs.append(waveform)
        return _make_model(graph)

    def _build_encoder(
        self,
        encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build encoder: waveform → codes.

        Inputs:
            waveform: (B, 1, audio_samples) float32
        Outputs:
            codes: (B, num_quantizers, T) int64
        """
        batch = ir.SymbolicDim("batch")
        audio_len = ir.SymbolicDim("audio_length")

        waveform = ir.Value(
            name="waveform",
            shape=ir.Shape([batch, 1, audio_len]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, builder = _make_graph([waveform], name="encoder")
        codes = encoder(builder.op, waveform)

        codes.name = "codes"
        graph.outputs.append(codes)
        return _make_model(graph)
