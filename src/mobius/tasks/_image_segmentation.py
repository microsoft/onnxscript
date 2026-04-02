# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Image segmentation task for models that take both vision and text inputs."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ImageSegmentationTask(ModelTask):
    """Text-guided image segmentation (CLIPSeg-style).

    Inputs:
        - pixel_values: [batch, channels, height, width] FLOAT
        - input_ids: [batch, text_seq_len] INT64
        - attention_mask: [batch, text_seq_len] INT64

    Outputs:
        - logits: [batch, height_out, width_out] FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        text_seq_len = ir.SymbolicDim("text_seq_len")

        image_size = getattr(config, "image_size", 224)
        num_channels = getattr(config, "num_channels", 3)

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_channels, image_size, image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, text_seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, text_seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([pixel_values, input_ids, attention_mask])
        op = builder.op

        logits = module(
            op,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        return ModelPackage({"model": _make_model(graph)}, config=config)
