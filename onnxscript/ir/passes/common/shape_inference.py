# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shape inference pass using onnx.shape_inference."""

from __future__ import annotations

__all__ = [
    "ShapeInferencePass",
    "infer_shapes",
]

import functools
import logging

import onnx

from onnxscript import ir
from onnxscript.ir.passes.common import _c_api_utils

logger = logging.getLogger(__name__)


def _merge_func(model: ir.Model, inferred_proto: onnx.ModelProto) -> tuple[ir.Model, bool]:
    """Merge the inferred model with the original model.

    Args:
        model: The original IR model.
        inferred_proto: The inferred ONNX model.

    Returns:
        A tuple containing the modified model and a boolean indicating whether the model was modified.
    """
    inferred_model = ir.serde.deserialize_model(inferred_proto)
    modified = False
    for original_graph, inferred_graph in (model.graphs(), inferred_model.graphs()):
        original_values = ir.convenience.create_value_mapping(original_graph)
        inferred_values = ir.convenience.create_value_mapping(inferred_graph)
        for name, value in original_values.items():
            if name in inferred_values:
                inferred_value = inferred_values[name]
                if value.shape != inferred_value.shape and inferred_value.shape is not None:
                    value.shape = inferred_value.shape
                    modified = True
                if value.dtype != inferred_value.dtype and inferred_value.dtype is not None:
                    value.dtype = inferred_value.dtype
                    modified = True
            else:
                logger.warning(
                    "Value %s not found in inferred graph %s", name, inferred_graph.name
                )
    return model, modified


class ShapeInferencePass(ir.passes.InPlacePass):
    """This pass performs shape inference on the graph."""

    def __init__(
        self, check_type: bool = True, strict_mode: bool = True, data_prop: bool = True
    ) -> None:
        """Initialize the shape inference pass.

        Args:
            check_type: If True, check the types of the inputs and outputs.
            strict_mode: If True, use strict mode for shape inference.
            data_prop: If True, use data propagation for shape inference.
        """
        super().__init__()
        self.check_type = check_type
        self.strict_mode = strict_mode
        self.data_prop = data_prop

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        onnx_infer_shapes = functools.partial(
            onnx.shape_inference.infer_shapes,
            check_type=self.check_type,
            strict_mode=self.strict_mode,
            data_prop=self.data_prop,
        )

        inferred_model, modified = _c_api_utils.call_onnx_api(
            onnx_infer_shapes,
            model,
            merge_func=_merge_func,
        )

        return ir.passes.PassResult(inferred_model, modified=modified)


def infer_shapes(
    model: ir.Model,
    *,
    check_type: bool = True,
    strict_mode: bool = True,
    data_prop: bool = True,
) -> ir.Model:
    """Perform shape inference on the model.

    Args:
        model: The model to perform shape inference on.
        check_type: If True, check the types of the inputs and outputs.
        strict_mode: If True, use strict mode for shape inference.
        data_prop: If True, use data propagation for shape inference.

    Returns:
        The model with shape inference applied.
    """
    return ShapeInferencePass(
        check_type=check_type, strict_mode=strict_mode, data_prop=data_prop
    )(model).model
