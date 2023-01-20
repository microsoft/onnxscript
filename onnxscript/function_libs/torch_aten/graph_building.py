"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import collections
import typing
import warnings
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import onnx
import torch
from torch.onnx import _type_utils
from torch.onnx._internal import jit_utils

import onnxscript
from onnxscript import evaluator
from onnxscript import tensor as onnxscript_tensor

ValidArgumentType = Union["TorchScriptTensor", str, int, float]


class TorchScriptTensor(onnxscript_tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""

    def __init__(self, value: torch.Value):
        super().__init__(None)
        self._value = value

    @property
    def value(self) -> np.ndarray:
        return None

    def symbolic_value(self) -> torch.Value:
        return self._value

    @property
    def rank(self) -> int | None:
        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
        return value_type.dim()

    @property
    def shape(self) -> tuple[int | None, ...] | None:
        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
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


def _parse_torch_value(value: torch.Value, attr_type: onnx.AttributeProto.AttributeType):
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


def _parse_node(value: torch.Value):
    # Why do we find the node and then get the same value back?
    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == "onnx::Constant":
        return torch.onnx.symbolic_helper._node_get(node, "value")
    raise ValueError("[ERROR] Attribute is not Constant!!!")


def _split_args_kwargs_to_input_attr(onnx_func: onnxscript.OnnxFunction, args, kwargs):
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
        onnx_attrs[attr_proto.name] = attr_value

    for key, value in kwargs.items():
        if key not in attr_name_to_protos:
            warnings.warn(f"Attribute '{key}' is not defined in the function definition")
            continue
        # Fill in the values from kwargs
        onnx_attrs[key] = value

    # Fill in the default values
    for key, attr_proto in attr_name_to_protos.items():
        if key not in onnx_attrs:
            onnx_attrs[key] = attr_proto.value

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


def _unwrap_tensor_to_torch_value(
    value: Union[Dict[str, Any], List]
) -> Union[Dict[str, Any], List]:
    # unwrap TorchScriptTensor
    if isinstance(value, dict):
        value = {
            k: v.symbolic_value() if isinstance(v, TorchScriptTensor) else v
            for k, v in value.items()
        }
    elif isinstance(value, list):
        value = [v.symbolic_value() if isinstance(v, TorchScriptTensor) else v for v in value]
    elif isinstance(value, tuple):
        return tuple(v.symbolic_value() if isinstance(v, TorchScriptTensor) else v for v in value)
    elif isinstance(value, TorchScriptTensor):
        value = value.symbolic_value()
    else:
        print("value: ", value)
    # A normal python value
    return value


def _wrap_torch_value_to_tensor(
    value: Union[Dict[str, Any], List]
) -> Union[Dict[str, Any], List]:
    # wrap torch.Value with TorchScriptTensor
    if isinstance(value, dict):
        value = {
            k: TorchScriptTensor(v) if isinstance(v, torch.Value) else v
            for k, v in value.items()
        }
    elif isinstance(value, list):
        value = [TorchScriptTensor(v) if isinstance(v, torch.Value) else v for v in value]
    elif isinstance(value, tuple):
        return tuple(TorchScriptTensor(v) if isinstance(v, torch.Value) else v for v in value)
    elif isinstance(value, torch.Value):
        value = TorchScriptTensor(value)
    return value

def _unwrap_tensors_to_torch_values(tensors):
    if isinstance(tensors, Sequence):
        return [_unwrap_tensor_to_torch_value(output) for output in tensors]
    return _unwrap_tensor_to_torch_value(tensors)

class TorchScriptEvaluator(evaluator.Evaluator):
    def __init__(self, graph: TorchScriptGraph):
        self._graph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    def eval_function(self, function: onnxscript.OnnxFunction, args, kwargs):
        # args/kwargs are TorchScriptTensor/python built-in based
        inputs, attributes = _split_args_kwargs_to_input_attr(function, args, kwargs)
        return self._graph.add_function(function, inputs, attributes)

    def _eval(self, schema, inputs, attributes, closure):
        # TODO: Does it really know what the inputs are?
        return self._graph.add_op(schema, inputs, attributes)


class TorchScriptGraph:
    def __init__(self):
        self._graph = torch._C.Graph()
        self._graph_context = jit_utils.GraphContext(
            graph=self._graph,
            block=self._graph.block(),  # Pointless. Just make linter happy.
            opset=17,
            original_node=self._graph.insertPoint(),  # Pointless. Just make linter happy.
            params_dict={},  # Pointless. Just make linter happy.
            env={},  # Pointless. Just make linter happy.
        )
        # All the functions used, deduplicated by name
        self._function_store: dict[str, onnxscript.OnnxFunction] = {}

    @property
    def graph(self):
        return self._graph

    @property
    def graph_context(self):
        return self._graph_context

    def add_input(self, input_name: str, input_value: torch.Tensor) -> TorchScriptTensor:
        # TODO: Take in a TorchScriptTensor?
        if input_value is None:
            # This input argument is None, which is mapped
            # to a NULL value in TorchScript type system.
            torch_value = self.graph.op("prim::Constant")  # type: ignore[attr-defined]
            torch_value.setType(torch._C.OptionalType.ofTensor())
            return torch_value
        torch_value = self._graph.addInput(input_name)
        torch_value.setType(torch._C.TensorType.create_from_tensor(input_value))
        tensor_value = _wrap_torch_value_to_tensor(torch_value)
        return tensor_value

    def register_outputs(
        self, outputs: Union[TorchScriptTensor, tuple[TorchScriptTensor, ...]]
    ):
        unwrapped_outputs = _unwrap_tensors_to_torch_values(outputs)
        if isinstance(unwrapped_outputs, torch.Value):
            self._graph.registerOutput(unwrapped_outputs)
        else:
            for ts_output in unwrapped_outputs:
                assert isinstance(
                    ts_output, torch.Value
                ), f"ts_output must be a torch._C.Value, not {type(ts_output)}"
                self._graph.registerOutput(unwrap_outputs)
        return

    def _add_torchscript_op(
        self,
        name,
        onnx_inputs,
        onnx_attributes,
        outputs: int,
    ) -> TorchScriptTensor | tuple[TorchScriptTensor, ...]:
        
        unwrapped_inputs = _unwrap_tensors_to_torch_values(onnx_inputs)
        graph_inputs = []
        for input in unwrapped_inputs:
            if not isinstance(input, torch.Value):
                graph_inputs.append(self._wrap_constant_to_torchscript_value(input))
            else:
                graph_inputs.append(input)
        for value in onnx_attributes.values():
            assert not isinstance(value, TorchScriptTensor)
        encoded_attributes = _convert_kwargs_for_torchscript(onnx_attributes)
        result = self._graph_context.op(
            name, *graph_inputs, outputs=outputs, **encoded_attributes
        )
        if isinstance(result, Sequence):
            return tuple(TorchScriptTensor(v) for v in result)
        return TorchScriptTensor(result)

    def add_op(
        self,
        onnx_op_schema,
        onnx_inputs: Sequence[ValidArgumentType | Sequence[ValidArgumentType]],
        onnx_attributes: dict[str, ValidArgumentType | Sequence[ValidArgumentType]],
    ):
        # Compute outputs from the onnx_op op schema

        # FIXME(justinchuby): Figure out how to get the number of outputs from the schema
        result = self._add_torchscript_op(
            f"onnx::{onnx_op_schema.name}", onnx_inputs, onnx_attributes, outputs=1
        )

        return result

    def add_function(
        self,
        onnx_function: onnxscript.OnnxFunction,
        onnx_inputs,
        onnx_attributes,
    ):
        self._function_store[onnx_function.name] = onnx_function

        # Compute outputs from the function schema

        result = self._add_torchscript_op(
            f"{onnx_function.function_ir.domain}::{onnx_function.name}", onnx_inputs, onnx_attributes, outputs=len(onnx_function.function_ir.outputs)
        )

        return result

    def _wrap_constant_to_torchscript_value(self, constant) -> torch.Value:
        if isinstance(constant, float):
            # Always promote scalar to tensor with element type "dtype."
            # Usually, "dtype" is extracted from the expected output tensor of the node.
            # If this assumption is broken, we probably need to
            #  1. add "scalar" type in ONNX  and extend all exporters to support it, or
            #  2. write type promotion logic for each operator.
            # TODO(wechi): the called exporting function should tell all allowed input and output types.
            # Then, here we can try type-casting if type-mismatch happens.
            value = self.graph.op(
                "Constant", value_t=torch.tensor(constant, dtype=torch.float)
            )
        elif isinstance(constant, int):
            # Always promote scalar to tensor with element type "dtype."
            # Usually, "dtype" is extracted from the expected output tensor of the node.
            # If this assumption is broken, we probably need to
            #  1. add "scalar" type in ONNX  and extend all exporters to support it, or
            #  2. write type promotion logic for each operator.
            # TODO(wechi): the called exporting function should tell all allowed input and output types.
            # Then, here we can try type-casting if type-mismatch happens.
            value = self.graph.op(
                "Constant", value_t=torch.tensor(constant, dtype=torch.int64)
            )
        elif constant is None:
            value = self.graph.op("prim::Constant")
            value.setType(torch._C.OptionalType.ofTensor())
        elif isinstance(constant, list) and all(isinstance(val, int) for val in constant):
            value = self.graph.op(
                "Constant", value_t=torch.tensor(constant, dtype=torch.int64)
            )
        elif isinstance(constant, list) and all(isinstance(val, float) for val in constant):
            value = self.graph.op(
                "Constant", value_t=torch.tensor(constant, dtype=torch.float)
            )
        else:
            raise ValueError("[ERROR]: ", constant)
        return value

    def to_model_proto(
        self, initializers: dict[str, torch.Tensor], opset_version: Optional[int]
    ):
        proto, _, _, _ = self.graph._export_onnx(
            initializers=initializers,
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=False,
            custom_opsets={},
            add_node_names=True,
            onnx_file_path="",
            node_attr_to_name={},
        )

        onnx_model = onnx.load_from_string(proto)
        function_proto_list = []
        for onnx_function in self._function_store.values():
            function_proto_list.append(onnx_function.to_function_proto())
        onnx_model.functions.extend(function_proto_list)
        print("ONNX model: \n", onnx_model)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        print("after ONNX model: \n", onnx_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        print("[Success] ONNX model")
        model_bytes = onnx_model.SerializeToString()
        return model_bytes
