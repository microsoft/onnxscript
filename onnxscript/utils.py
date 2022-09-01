# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numbers
import numpy as np
from typing import Any, Sequence
import onnx
from onnx import TensorProto, ValueInfoProto, ModelProto, FunctionProto
from onnx.helper import make_tensor_type_proto, make_sequence_type_proto

# print utility unavailable in ONNX 1.12 or earlier:
try:
    from onnx.printer import to_text as proto2text
except ImportError:
    def proto2text(x):
        return "<print utility unavailable>"


def value_to_type_proto(val):
    '''
    Return the ONNX type of a python-value.
    '''
    if isinstance(val, np.ndarray):
        elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
        shape = val.shape
        return make_tensor_type_proto(elem_type, shape)
    if isinstance(val, numbers.Number):
        nparray = np.array(val)
        elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[nparray.dtype]
        return make_tensor_type_proto(elem_type, [])
    if isinstance(val, list):
        if len(val) > 0:
            return make_sequence_type_proto(value_to_type_proto(val[0]))
        else:
            # Edge-case. Cannot determine a suitable ONNX type for an empty list.
            # Should be using a typed-value instead.
            # Treated as a sequence of tensors of float-type.
            return make_sequence_type_proto(make_tensor_type_proto(TensorProto.FLOAT, None))
    else:
        raise ValueError(
            f"Cannot convert a {type(val)} to TypeProto")


def values_to_value_infos(names, values):
    '''
    Create a list of ValueInfoProto representing a list of names and a corresponding
    list of values, skipping any None values.
    '''
    return [onnx.helper.make_value_info(name, value_to_type_proto(val))
            for (name, val) in zip(names, values)
            if val is not None]


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
