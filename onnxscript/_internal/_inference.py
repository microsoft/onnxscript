# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable

import numpy as np
import onnx
import onnx_ir as ir

def _get_numpy_value(
    val: ir.Value | None, dtype: ir.DataType | None = None, size_limit: int | None = None
) -> np.ndarray | None:
    """Returns the numpy value of a constant value, if available.

    It returns None if the value is not a constant value, or if the value is not of
    the specified element dtype, or if the size of the value exceeds the specified
    size_limit.
    """
    if val is None:
        return None
    const_value = val.const_value
    if const_value is not None:
        if dtype is not None and const_value.dtype != dtype:
            return None
        if size_limit is not None and const_value.size > size_limit:
            return None
        try:
            # Turn the constant value into a numpy array representation with the
            # specifics of this conversion handled by the tensor type
            array = const_value.numpy()
            # Can/should not reinterpret strings via .view, resulting in
            #   "TypeError: Cannot change data-type for array of references."
            # There is also no reason to reinterpret strings, this is only
            # relevant for some arithmetic types
            if const_value.dtype != ir.DataType.STRING:
                # Reinterpret the array with `.view()` because some
                # implementations  of ir.TensorProtocol (e.g. PyTorch<=2.7) do
                # not use ml_dtypes for bfloat16 etc.
                array = array.view(const_value.dtype.numpy())
        except FileNotFoundError:
            # External data is not available.
            # logger.warning(
            #     "External data for value '%s' is not available. "
            #     "This may lead to incorrect constant folding.",
            #     val.name,
            # )
            return None
        assert isinstance(array, np.ndarray)
        return array
    return None

def _do_onnx_inference(node: ir.Node, opset_version: int) -> None:
    output_types = {}

    def get_constant_value(x: ir.Value) -> onnx.TensorProto | None:
        value = _get_numpy_value(x, size_limit=20)
        if value is not None:
            assert x.const_value is not None
            return ir.serde.serialize_tensor(x.const_value)
        return None

    def get_type(index: int, value: ir.Value) -> onnx.TypeProto:
        if value.type is None:
            raise ValueError(f"Type of input {index} value {value.name} of node {node.name} not known")
        type_proto = ir.serde.serialize_type(value.type)
        if value.shape is not None:
            ir.serde.serialize_shape_into(type_proto, value.shape)
        return type_proto

    input_types = {x.name: get_type(i, x) for i, x in enumerate(node.inputs) if x is not None}
    input_data = {x.name: get_constant_value(x) for x in node.inputs if x is not None}
    input_data = {k: v for k, v in input_data.items() if v is not None}

    # TODO: pass in constant values, ir_version
    schema = onnx.defs.get_schema(
        node.op_type, opset_version, node.domain
    )
    output_types = onnx.shape_inference.infer_node_outputs(
        schema,
        ir.serde.serialize_node(node),
        input_types,  # type: ignore[arg-type]
        input_data,  # type: ignore[arg-type]
    )
    for output in node.outputs:
        if output.name in output_types:
            inferred_type = output_types[output.name]
            # TODO: merge types, check for conflicts
            inferred_shape = ir.serde.deserialize_type_proto_for_shape(
                inferred_type
            )
            # NOTE: forward shape inference
            output.merge_shapes(inferred_shape)
            output.type = ir.serde.deserialize_type_proto_for_type(inferred_type)


def infer_outputs(node: ir.Node, opset_version: int) -> None:
    try:
        _do_onnx_inference(node, opset_version)
    except Exception as e:
        # TODO: compose with any existing error
        node.metadata_props["inference_error"] = str(e)
