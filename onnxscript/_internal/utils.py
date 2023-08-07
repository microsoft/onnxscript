# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import numbers
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import onnx
import onnx.helper
from onnx import FunctionProto, ModelProto, TensorProto, ValueInfoProto

from onnxscript import tensor

# print utility unavailable in ONNX 1.12 or earlier:
# pylint: disable=unused-import, ungrouped-imports
try:
    from onnx.printer import to_text as proto2text
except ImportError:

    def proto2text(_: Any) -> str:  # type: ignore[misc]
        return "<print utility unavailable>"


# pylint: enable=unused-import, ungrouped-imports


def external_tensor(
    name: str,
    data_type: int,
    dims: Sequence[int],
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> TensorProto:
    """Create a TensorProto referencing externally stored tensor-data.

    Args:
        name: name of the tensor
        data_type: data type of tensor element
        dims: shape of the tensor
        location: location of the external file (relative path)
        offset: offset in the file where the tensor-data starts
        length: number of bytes containing the data
        checksum: SHA1 digest of the file
        basepath: basepath combined with location to form the full path

    Returns:
        TensorProto

    See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md for more details.
    """
    tensor_proto = TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = data_type
    tensor_proto.dims.extend(dims)
    tensor_proto.data_location = TensorProto.EXTERNAL

    def add(k, v):
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    add("location", location)
    if offset is not None:
        add("offset", int(offset))
    if length is not None:
        add("length", int(length))
    if checksum is not None:
        add("checksum", checksum)
    if basepath is not None:
        add("basepath", basepath)
    return tensor_proto


def value_to_type_proto(val):
    """Return the ONNX type of a python-value."""
    if isinstance(val, (np.ndarray, tensor.Tensor)):
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(val.dtype)
        shape = val.shape
        return onnx.helper.make_tensor_type_proto(elem_type, shape)
    if isinstance(val, int):
        return onnx.helper.make_tensor_type_proto(TensorProto.INT32, [])
    if isinstance(val, (float, np.float32)):
        return onnx.helper.make_tensor_type_proto(TensorProto.FLOAT, [])
    if isinstance(val, list):
        if len(val) > 0:
            return onnx.helper.make_sequence_type_proto(value_to_type_proto(val[0]))
        # Edge-case. Cannot determine a suitable ONNX type for an empty list.
        # Should be using a typed-value instead.
        # Treated as a sequence of tensors of float-type.
        return onnx.helper.make_sequence_type_proto(
            onnx.helper.make_tensor_type_proto(TensorProto.FLOAT, None)
        )
    if isinstance(val, numbers.Number):
        nparray = np.array(val)
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(nparray.dtype)
        return onnx.helper.make_tensor_type_proto(elem_type, [])
    raise ValueError(f"Value of type {type(val)} is invalid as an ONNX input/output.")


def values_to_value_infos(name_values):
    """Create a list of ValueInfoProto from a list of (name, value) pairs,
    skipping any None values.
    """
    return [
        onnx.helper.make_value_info(name, value_to_type_proto(val))
        for (name, val) in name_values
        if val is not None
    ]


def make_model_from_function_proto(
    function_proto: FunctionProto,
    function_opset_version: int,
    input_value_infos: Sequence[ValueInfoProto],
    output_value_infos: Sequence[ValueInfoProto],
    **attrs: Any,
) -> ModelProto:
    """Creates a model containing a single call to a given
    function with input and output value_infos, etc.

    Args:
        function_proto (FunctionProto): function proto
            representing a single call
        function_opset_version (int):  function_proto's version
        input_value_infos (list of ValueInfoProto): function's input
        output_value_infos (list of ValueInfoProto): function's output
        **attrs (dict): the attributes of the node for the function

    Returns:
        ModelProto
    """

    input_names = [vi.name for vi in input_value_infos]
    output_names = [vi.name for vi in output_value_infos]
    node = onnx.helper.make_node(
        function_proto.name,
        input_names,
        output_names,
        domain=function_proto.domain,
        **attrs,
    )
    graph = onnx.helper.make_graph([node], "node_graph", input_value_infos, output_value_infos)
    model_proto_opset: Iterable[onnx.OperatorSetIdProto] = function_proto.opset_import
    if all(o.domain != function_proto.domain for o in model_proto_opset):
        model_proto_opset = [
            *model_proto_opset,
            onnx.helper.make_opsetid(function_proto.domain, function_opset_version),
        ]
    model = onnx.helper.make_model(
        graph,
        functions=[function_proto],
        producer_name="onnxscript",
        opset_imports=model_proto_opset,
    )
    return model
