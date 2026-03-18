# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapter task for T2I-Adapter and IP-Adapter models."""

from __future__ import annotations

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class AdapterTask(ModelTask):
    """Build ONNX graph for conditioning adapters (T2I, IP-Adapter)."""

    name = "adapter"

    def build(
        self,
        module,
        config,
    ) -> ModelPackage:
        # Determine input shape based on adapter type
        if hasattr(config, "in_channels"):
            # T2I-Adapter: conditioning image input
            condition = ir.Value(
                name="condition",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape(("batch", config.in_channels, "height", "width")),
            )
        else:
            # IP-Adapter: image embedding input
            condition = ir.Value(
                name="image_embeds",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape(("batch", config.image_embed_dim)),
            )

        graph, builder = _make_graph([condition])
        op = builder.op

        outputs = module(op, condition)

        if isinstance(outputs, list):
            for i, out in enumerate(outputs):
                out.name = f"feature_{i}"
                graph.outputs.append(out)
        else:
            outputs.name = "adapter_output"
            graph.outputs.append(outputs)

        return ModelPackage({"model": _make_model(graph)})
