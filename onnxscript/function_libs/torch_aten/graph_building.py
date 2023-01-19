"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import collections
import typing
import warnings

import numpy as np
import onnx
import torch
from torch.onnx import _type_utils
from torch.onnx._internal import jit_utils

import onnxscript
from onnxscript import evaluator, tensor as onnxscript_tensor


class TorchScriptTensor(onnxscript_tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""

    def __init__(self, value: torch.Value):
        super().__init__(None)
        self._value = value

    @property
    def value(self)-> np.ndarray:
        raise NotImplementedError()

    def symbolic_value(self) -> torch.Value:
        return self._value

    @property
    def rank(self) -> int | None:
        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch._C.TensorType, value_type)
        return value_type.dim()

    @property
    def shape(self) -> tuple[int | None, ...] | None:
        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch._C.TensorType, value_type)
        shape = value_type.varyingSizes()
        if shape is None:
            return None
        return tuple(shape)

    @property
    def dtype(self):
        # TODO: Return numpy dtype
        return _type_utils.JitScalarType.from_value(
            self._value, default=_type_utils.JitScalarType.UNDEFINED
        ).dtype()

    @property
    def onnx_dtype(self):
        return _type_utils.JitScalarType.from_value(
            self._value, _type_utils.JitScalarType.UNDEFINED
        ).onnx_type()


def _parse_torch_value(value: torch.Value, attr_type):
    if attr_type == onnx.AttributeProto.FLOAT:
        return float(value)
    if attr_type == onnx.AttributeProto.INT:
        return int(value)
    if attr_type == onnx.AttributeProto.STRING:
        return str(value)
    if attr_type == onnx.AttributeProto.FLOATS:
        return [float(v) for v in value]
    if attr_type == onnx.AttributeProto.INTS:
        return [int(v) for v in value]

    return value


def _parse_node(value: torch._C.Value):
    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == "onnx::Constant":
        return torch.onnx.symbolic_helper._node_get(node, "value")
    raise ValueError("[ERROR] Attribute is not Constant!!!")


def adapt_torchscript_inputs(onnx_func, args, kwargs):
    func_ir = onnx_func.function_ir
    assert len(func_ir.inputs) + len(func_ir.attr_protos) == len(args)
    # The first len(func_ir.inputs) arguements are onnx inputs
    onnx_inputs = args[: len(func_ir.inputs)]
    # The rest is onnx attributes
    # Contruct a dictionary of attributes with names specified in the function
    # definition
    attributes = args[len(func_ir.inputs) :]
    onnx_attrs = {}

    # (1) Some/All attributes are supplied as positional arguments
    # (2) Some attributes are supplied as kwargs
    # (3) Some arguments in kwargs are not defined in the onnx function
    attr_name_to_protos = collections.OrderedDict(
        (attr.name, attr) for attr in func_ir.attr_protos
    )
    assert len(attr_name_to_protos) >= len(attributes)
    for attr_proto, attr_value in zip(attr_name_to_protos.values(), attributes):
        node_val = _parse_node(attr_value)
        onnx_attrs[attr_proto.name] = _parse_torch_value(node_val, attr_proto.type)

    for key, value in kwargs.items():
        if key not in attr_name_to_protos:
            warnings.warn(f"Attribute '{key}' is not defined in the function definition")
            continue
        # Fill in the values from kwargs
        attr_proto = attr_name_to_protos[key]
        onnx_attrs[key] = _parse_torch_value(value, attr_proto.type)

    # Fill in the default values
    for key, attr_proto in attr_name_to_protos.items():
        if key not in onnx_attrs:
            onnx_attrs[key] = attr_proto.value

    # wrap torch.Value with TorchScriptTensor
    onnx_inputs = [
        TorchScriptTensor(v) if isinstance(v, torch.Value) else v for v in onnx_inputs
    ]
    onnx_attrs = {
        k: TorchScriptTensor(v) if isinstance(v, torch.Value) else v
        for k, v in onnx_attrs.items()
    }

    return onnx_inputs, onnx_attrs


def _convert_kwargs_for_torchscript(kwargs):
    encoded = {}
    for attr_name, attr in kwargs.items():
        if isinstance(attr, float):
            attr_name += "_f"
        elif isinstance(attr, int):
            attr_name += "_i"
        elif isinstance(attr, str):
            attr_name += "_s"
        elif isinstance(attr, list):
            if isinstance(attr, float):
                attr_name += "_f"
            elif isinstance(attr, int):
                attr_name += "_i"
        encoded[attr_name] = attr
    return encoded


def _convert_result_to_torchscript(result):
    if isinstance(result, tuple):
        return tuple(v.symbolic_value() for v in result)
    return result.symbolic_value()


class TorchScriptEvaluator(evaluator.Evaluator):
    def __init__(self, graph: jit_utils.GraphContext):
        self._graph = graph
        self._ops_to_function = {}

    @property
    def graph(self):
        return self._graph

    def functions(self):
        return self._ops_to_function

    def eval_function(self, function: onnxscript.OnnxFunction, *args, **kwargs):
        self._ops_to_function[function.name] = function
        opname = function.opset.domain + "::" + function.name

        # unwrap TorchScriptTensor
        args = [arg.symbolic_value() if isinstance(arg, TorchScriptTensor) else arg for arg in args]

        kwargs = {
            k: v.symbolic_value() if isinstance(v, TorchScriptTensor) else v
            for k, v in kwargs.items()
        }
        encoded_kwargs = _convert_kwargs_for_torchscript(kwargs)

        # This is not a tuple for now. TODO: Check output
        result = self._graph.op(opname, *args, **encoded_kwargs)
        if isinstance(result, tuple):
            return tuple(TorchScriptTensor(v) for v in result)
        return TorchScriptTensor(result)

    def _eval(self, schema, inputs, attributes):
        return self._graph.op(schema.name, *inputs, **attributes)
