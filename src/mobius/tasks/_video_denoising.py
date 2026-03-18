# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Video denoising task for 3D diffusion models (CogVideoX, etc.).

Builds an ONNX graph that takes a 5D noisy video latent
[B, T, C, H, W] + timestep + text conditioning and produces
noise prediction of the same shape.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._diffusers_configs import CogVideoXConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class VideoDenoisingTask(ModelTask):
    """Build ONNX graph for video diffusion denoising."""

    name = "video-denoising"

    def build(
        self,
        module,
        config: CogVideoXConfig,
    ) -> ModelPackage:
        sample = ir.Value(
            name="sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(
                (
                    "batch",
                    "num_frames",
                    config.in_channels,
                    "height",
                    "width",
                )
            ),
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
