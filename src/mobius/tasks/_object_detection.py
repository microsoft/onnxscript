# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Object detection task for models like YOLOS and DETR."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ObjectDetectionTask(ModelTask):
    """Object detection with pixel_values input.

    Inputs:
        - pixel_values: [batch, channels, height, width] FLOAT

    Outputs:
        - logits: [batch, num_queries, num_classes] FLOAT
        - pred_boxes: [batch, num_queries, 4] FLOAT
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

        logits, pred_boxes = module(op, pixel_values=pixel_values)

        logits.name = "logits"
        pred_boxes.name = "pred_boxes"
        graph.outputs.append(logits)
        graph.outputs.append(pred_boxes)

        return ModelPackage({"model": _make_model(graph)}, config=config)
