# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Image classification task for ViT-like models."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ImageClassificationTask(ModelTask):
    """Image classification with pixel_values input.

    Inputs:
        - pixel_values: [batch, channels, height, width] FLOAT

    Outputs:
        - last_hidden_state: [batch, sequence_len, hidden_size] FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")

        image_size = getattr(config, "image_size", 224)
        num_channels = getattr(config, "num_channels", 3)

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_channels, image_size, image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, builder = _make_graph([pixel_values])
        op = builder.op

        last_hidden_state = module(op, pixel_values=pixel_values)

        last_hidden_state.name = "last_hidden_state"
        graph.outputs.append(last_hidden_state)

        return ModelPackage({"model": _make_model(graph)}, config=config)
