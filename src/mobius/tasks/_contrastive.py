# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Contrastive task for dual-encoder models (CLIP, CLAP).

Builds two separate ONNX models from a contrastive module:
1. **text**: input_ids + attention_mask → embeddings
2. **vision** (or **audio**): pixel_values (or input_features) → embeddings

The module must provide two sub-modules:

- ``text_encoder``: text encoder with
  ``forward(op, input_ids, attention_mask) → embeddings``
- ``modality_encoder``: vision/audio encoder with
  ``forward(op, pixel_values=...) → embeddings``
  or ``forward(op, input_features=...) → embeddings``

The ``modality`` attribute on the module (default ``"vision"``) controls
input naming and the key in the returned :class:`ModelPackage`.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ContrastiveTask(ModelTask):
    """Dual-encoder contrastive task (CLIP, CLAP, SigLIP, etc.).

    Produces two ONNX models: ``text`` and ``vision`` (or ``audio``).
    Each model independently maps its input to an embedding vector in a
    shared embedding space, enabling similarity search.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        modality: str = getattr(module, "modality", "vision")

        models: dict[str, ir.Model] = {}
        models["text"] = self._build_text(module.text_encoder, config)
        models[modality] = self._build_modality(module.modality_encoder, config, modality)

        return ModelPackage(models, config=config)

    # ------------------------------------------------------------------
    # Text encoder
    # ------------------------------------------------------------------

    def _build_text(
        self,
        text_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build text encoder: input_ids + attention_mask → embeddings."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([input_ids, attention_mask])
        op = builder.op

        text_embeds = text_encoder(op, input_ids=input_ids, attention_mask=attention_mask)

        text_embeds.name = "text_embeds"
        graph.outputs.append(text_embeds)
        return _make_model(graph)

    # ------------------------------------------------------------------
    # Vision / audio encoder
    # ------------------------------------------------------------------

    def _build_modality(
        self,
        modality_encoder: nn.Module,
        config: ArchitectureConfig,
        modality: str,
    ) -> ir.Model:
        """Build vision or audio encoder → embeddings."""
        batch = ir.SymbolicDim("batch")

        if modality == "vision":
            image_size = getattr(config, "image_size", 224)
            num_channels = getattr(config, "num_channels", 3)
            modality_input = ir.Value(
                name="pixel_values",
                shape=ir.Shape([batch, num_channels, image_size, image_size]),
                type=ir.TensorType(ir.DataType.FLOAT),
            )
        elif modality == "audio":
            modality_input = ir.Value(
                name="input_features",
                shape=ir.Shape([batch, "time", "freq"]),
                type=ir.TensorType(ir.DataType.FLOAT),
            )
        else:
            raise ValueError(f"Unsupported modality: {modality!r}")

        graph, builder = _make_graph([modality_input])
        op = builder.op

        embeds = modality_encoder(op, modality_input)

        embeds.name = f"{modality}_embeds"
        graph.outputs.append(embeds)
        return _make_model(graph)
