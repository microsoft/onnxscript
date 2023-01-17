"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import onnx
import torch
from torch.onnx import _type_utils
from torch.onnx._internal import jit_utils


import onnxscript
from onnxscript import evaluator

class TorchScriptTensor(onnxscript.tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""
    def __init__(self, value: torch.Value):
        super().__init__(None)
        self._value = value

    @property
    def value(self) -> torch.Value:
        return self._value

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        # TODO: Return numpy dtype
        return _type_utils.JitScalarType.from_value(
            self._value, _type_utils.JitScalarType.UNDEFINED
        ).dtype()

    @property
    def onnx_dtype(self):
        return _type_utils.JitScalarType.from_value(
            self._value, _type_utils.JitScalarType.UNDEFINED
        ).onnx_type()



class TorchScriptEvaluator(evaluator.Evaluator):
    def __init__(self, graph: jit_utils.GraphContext):
        self._graph = graph
        self._ops_to_function = {}

    def get_functions(self):
        return self._ops_to_function

    def get_graph(self):
        return self._graph

    def update_graph(self, graph):
        self._graph = graph

    def reset_graph(self):
        self._graph = None

    def _parse_node(self, value: torch._C.Value):
        node = value.node()
        if node.mustBeNone():
            return None
        if node.kind() == "onnx::Constant":
            return torch.onnx.symbolic_helper._node_get(node, "value")
        raise ValueError("[ERROR] Attribute is not Constant!!!")

    def decode_attributes(self, onnx_func, args, kwargs):
        func_ir = onnx_func.function_ir
        print("kwargs: ", kwargs)
        print("args: ", args)
        print("func_ir.inputs: ", func_ir.inputs)
        assert len(func_ir.inputs) + len(func_ir.attr_protos) == len(args)
        # The first len(func_ir.inputs) arguements are onnx inputs
        onnx_inputs = args[:len(func_ir.inputs)]
        # The rest is onnx attributes
        # Contruct a dictionary of attributes with names specified in the function
        # definition
        attributes = args[len(func_ir.inputs):]
        onnx_attrs = {}
        for attr in func_ir.attr_protos:
            if attr.name in func_ir.attrs:
                # with user given value
                attr_index = func_ir.attrs.index(attr.name)
                attr_value = attributes[attr_index]
                node_val = self._parse_node(attr_value)
            else:
                node_val = attr.value
            if attr.type == onnx.AttributeProto.FLOAT:
                onnx_attrs[attr.name] = float(node_val)
            elif attr.type == onnx.AttributeProto.INT:
                onnx_attrs[attr.name] = int(node_val)
            elif attr.type == onnx.AttributeProto.STRING:
                onnx_attrs[attr.name] = str(node_val)
            elif attr.type == onnx.AttributeProto.FLOATS:
                onnx_attrs[attr.name] = [float(v) for v in node_val]
            elif attr.type == onnx.AttributeProto.INTS:
                onnx_attrs[attr.name] = [int(v) for v in node_val]
            elif attr.type == onnx.AttributeProto.STRINGS:
                assert False, "Bad: list of strings"

        return onnx_inputs, onnx_attrs

    def _encode_kwargs(self, kwargs):
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


    def eval_function(self, function: onnxscript.OnnxFunction, *args, **kwargs):
        self._ops_to_function[function.name] = function
        opname = function.opset.domain + "::" + function.name

        args = [arg.value for arg in args if isinstance(arg, TorchScriptTensor)]
        encoded_kwargs = self._encode_kwargs(kwargs)
        print("encoded_kwargs:", encoded_kwargs)

        # This is not a tuple for now. TODO: Check output
        result = self._graph.op(opname, *args, **encoded_kwargs)
        if isinstance(result, tuple):
            return tuple(TorchScriptTensor(v) for v in result)
        return TorchScriptTensor(result)


    def _eval(self, schema, inputs, attributes):
        return self._graph.op(schema.name, *inputs, **attributes)
