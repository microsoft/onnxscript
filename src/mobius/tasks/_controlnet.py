# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ControlNet task: produces residuals for UNet conditioning."""

from __future__ import annotations

import onnx_ir as ir

from mobius._model_package import ModelPackage

# ControlNetConfig lives in the model file because diffusion models use
# their own config types (from_diffusers) rather than ArchitectureConfig.
# Tasks depend on models, so this import direction is correct.
from mobius.models.controlnet import ControlNetConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ControlNetTask(ModelTask):
    """Build ONNX graph for ControlNet residual generation."""

    name = "controlnet"

    def build(
        self,
        module,
        config: ControlNetConfig,
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
        controlnet_cond = ir.Value(
            name="controlnet_cond",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(
                ("batch", config.conditioning_channels, "cond_height", "cond_width")
            ),
        )

        graph, builder = _make_graph(
            [sample, timestep, encoder_hidden_states, controlnet_cond]
        )
        op = builder.op

        down_outputs, mid_output = module(
            op,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

        # Register outputs
        for i, out in enumerate(down_outputs):
            out.name = f"down_block_res_{i}"
            graph.outputs.append(out)
        mid_output.name = "mid_block_res"
        graph.outputs.append(mid_output)

        return ModelPackage({"model": _make_model(graph)})
