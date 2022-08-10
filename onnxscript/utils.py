# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numbers
import numpy as np
from typing import Any, Sequence
import onnx
from onnx import TensorProto, ValueInfoProto, ModelProto, FunctionProto
from .eager_numpy import NumpyArray


# print utility unavailable in ONNX 1.12 or earlier:
try:
    from onnx.printer import to_text as proto_to_text
except ImportError:
    def proto_to_text(
        x): return "<print utility unavailable>"


def map_pytype_to_schema_allowed_dtype(onnx_schema_types, dtype):
    # ONNX TensorProto data type is a supper set of python dtype.
    # When a dtype is not allowed by ONNX schema, we need to find a closest
    # dtype allowed by the schema.
    if dtype == 'int32':
        if 'tensor(int32)' not in onnx_schema_types and\
                'tensor(int64)' in onnx_schema_types:
            return np.dtype('int64')
    return dtype


def convert_arrays_to_value_infos(names, arr_list, op_schema_formal_parameter=None):
    if op_schema_formal_parameter is None:
        op_schema_formal_parameter = []

    value_infos = []
    for i, (name, arr) in enumerate(zip(names, arr_list)):
        elem_type: TensorProto.DataType
        shape: tuple

        if isinstance(arr, list):
            # sequence, assuming it is a float sequence
            # list should be replace by another container retaining the type information
            nparray = np.asarray(
                arr)
            if len(arr) == 0:
                nparray = nparray.astype(
                    np.float32)
            if op_schema_formal_parameter and len(op_schema_formal_parameter) > i:
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                    map_pytype_to_schema_allowed_dtype(
                        op_schema_formal_parameter[i].types, nparray.dtype)]
            else:
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[nparray.dtype]
            info = onnx.helper.make_tensor_sequence_value_info(
                name=name, elem_type=elem_type, shape=None)
            value_infos.append(
                info)
            continue

        if isinstance(arr, NumpyArray):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
            shape = arr.shape
        elif isinstance(arr, numbers.Number):
            nparray = np.array(arr)
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[nparray.dtype]
            shape = nparray.shape
        elif arr is None:
            continue
        else:
            raise ValueError(
                f"Cannot convert a {type(arr)} to value_info")

        value_info = onnx.helper.make_tensor_value_info(
            name=name,
            elem_type=elem_type,
            shape=shape)
        value_infos.append(value_info)
    return value_infos


def make_model_from_function_proto(
        function_proto: FunctionProto,
        function_opset_version: int,
        input_value_infos: Sequence[ValueInfoProto],
        output_value_infos: Sequence[ValueInfoProto],
        **attrs: Any
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

    input_names = [
        vi.name for vi in input_value_infos]
    output_names = [
        vi.name for vi in output_value_infos]
    node = onnx.helper.make_node(
        function_proto.name, input_names, output_names,
        domain=function_proto.domain,
        **(attrs or {}))
    graph = onnx.helper.make_graph(
        [node], "node_graph",
        input_value_infos, output_value_infos)
    model_proto_opset = function_proto.opset_import
    if all(o.domain != function_proto.domain for o in model_proto_opset):
        model_proto_opset = [
            *model_proto_opset,
            onnx.helper.make_opsetid(function_proto.domain, function_opset_version)]
    model = onnx.helper.make_model(
        graph,
        functions=[
            function_proto],
        producer_name='onnx-script',
        opset_imports=model_proto_opset)
    return model
