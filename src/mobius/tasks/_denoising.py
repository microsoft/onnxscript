# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Denoising task for diffusion models (UNet, DiT, etc.).

Builds an ONNX graph that takes noisy latent + timestep + conditioning
and produces noise prediction.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._diffusers_configs import UNet2DConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class DenoisingTask(ModelTask):
    """Build ONNX graph for diffusion denoising."""

    name = "denoising"

    def build(
        self,
        module,
        config: UNet2DConfig,
    ) -> ModelPackage:
        sample = ir.Value(
            name="sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", config.in_channels, "height", "width")),
        )
        timestep = ir.Value(
            name="timestep",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(("batch",)),
        )
        encoder_hidden_states = ir.Value(
            name="encoder_hidden_states",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", "sequence_length", config.cross_attention_dim)),
        )

        graph, builder = _make_graph([sample, timestep, encoder_hidden_states])
        op = builder.op

        noise_pred = module(
            op,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        noise_pred.name = "noise_pred"
        graph.outputs.append(noise_pred)

        return ModelPackage({"model": _make_model(graph)})
