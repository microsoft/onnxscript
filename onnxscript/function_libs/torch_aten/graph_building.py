"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import collections
import typing
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from typing_extensions import TypeAlias

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.shape_inference
import torch
from beartype import beartype
from torch.onnx import _type_utils

import onnxscript
from onnxscript import evaluator
from onnxscript import tensor as onnxscript_tensor
from onnxscript.function_libs.torch_aten import param_manipulation

__all__ = [
    "TorchScriptTensor",
    "TorchScriptGraph",
    "TorchScriptTracingEvaluator",
]


ValidArgumentType: TypeAlias = Union[
    "TorchScriptTensor",
    Sequence["TorchScriptTensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidInputType: TypeAlias = Union[
    "TorchScriptTensor",
    Sequence["TorchScriptTensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidTorchValueType: TypeAlias = Union[
    torch.Value,
    Sequence[torch.Value],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]

# TODO(justinchuby): Build a context manager to handle source information.


class TorchScriptTensor(onnxscript_tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""


    def __init__(self, value: torch.Value):
        super().__init__(None)
        self._value = value
        self._shape = None

    @property
    def value(self) -> np.ndarray:
        return None

    @property
    def rank(self) -> int | None:
        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
        return value_type.dim()

    @property
    def shape(self) -> Tuple[int | None, ...] | None:
        if self._shape is not None:
            return self._shape

        value_type = self._value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
        shape = value_type.varyingSizes()
        if shape is None:
            return None
        return tuple(shape)

    @shape.setter
    def shape(self, shape: Tuple[int | None, ...]):
        self._shape = shape
        # TODO(justinchuby): Add shape to self._value?

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

    def symbolic_value(self) -> torch.Value:
        """The symbolic Value in torch.Graph."""
        return self._value


@beartype
def _unwrap_tensor_to_torch_value(
    value: Union[ValidArgumentType, Dict[str, ValidArgumentType], Sequence[ValidArgumentType]]
) -> Union[ValidTorchValueType, Dict[str, ValidTorchValueType], List[ValidTorchValueType], Tuple[
    ValidTorchValueType, ...
]]:
    """Unwrap the TorchScriptTensor to torch.Value."""
    if isinstance(value, TorchScriptTensor):
        return value.symbolic_value()
    if isinstance(value, dict):
        return {k: _unwrap_tensor_to_torch_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_tensor_to_torch_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unwrap_tensor_to_torch_value(v) for v in value)

    # A normal python value
    return value


@beartype
def _wrap_torch_value_to_tensor(
    value: Union[torch.Value, Dict[str, ValidTorchValueType], Sequence[ValidTorchValueType]]
) -> Union[ValidArgumentType, Dict[str, ValidArgumentType], List[ValidArgumentType], Tuple[
    ValidArgumentType, ...
]]:
    """Wrap torch.Value to TorchScriptTensor."""
    if isinstance(value, torch.Value):
        return TorchScriptTensor(value)
    if isinstance(value, dict):
        return {k: _wrap_torch_value_to_tensor(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_wrap_torch_value_to_tensor(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap_torch_value_to_tensor(v) for v in value)

    return value


def _unwrap_tensors_to_torch_values(tensors):
    # TODO(justinchuby): Do we really need this?
    if isinstance(tensors, Sequence):
        return [_unwrap_tensor_to_torch_value(output) for output in tensors]
    return _unwrap_tensor_to_torch_value(tensors)


class TorchScriptTracingEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, graph: TorchScriptGraph):
        self._graph: TorchScriptGraph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    @beartype
    def eval_function(
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Dict[str, ValidArgumentType],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = param_manipulation.extract_param_schema_from_function(function)
        inputs, attributes = param_manipulation.separate_input_attributes_from_arguments(param_schemas, args, kwargs)
        return self._graph.add_function_call(function, inputs, attributes)

    def _eval(self, schema: onnx.defs.OpSchema, inputs, attributes, closure: Any):
        del closure  # Unused

        param_schemas = param_manipulation.extract_param_schema_from_op_schema(schema)
        inputs, attributes = param_manipulation.separate_input_attributes_from_arguments(param_schemas, inputs, attributes)
        return self._graph.add_op_call(schema, inputs, attributes)

    def eval(self, schema, inputs, attributes):
        outputs = self._eval(schema, inputs, attributes, closure=None)
        return outputs

@beartype
def _add_attribute_to_torchscript_node(
    node: torch.Node,
    key: str,
    value: Union[float, int, str, Sequence[float], Sequence[int], torch.Tensor],
):
    """Initializes the right attribute based on type of value."""
    if isinstance(value, float):
        return node.f_(key, value)
    if isinstance(value, int):
        return node.i_(key, value)
    if isinstance(value, str):
        return node.s_(key, value)
    if isinstance(value, torch.Tensor):
        return node.t_(key, value)
    if isinstance(value, Sequence):
        if isinstance(value[0], float):
            return node.fs_(key, list(value))
        if isinstance(value[0], int):
            # Need to use getattr because 'is' is a reserved keyword
            return getattr(node, "is_")(key, list(value))
        raise TypeError(f"Unsupported sequence type '{type(value)}' for attribute '{key}'")
    raise TypeError(f"Unsupported attribute type '{type(value)}' for attribute '{key}'")


@beartype
def _create_op_call_in_torch_graph(
    graph: torch.Graph,
    opname: str,
    *,
    inputs: Sequence[torch.Value],
    attributes: Dict[str, Any],
    n_outputs: int = 1,
) -> Tuple[torch.Value, ...]:
    """Creates a node representing an onnx op in `graph`.

    Args:
        graph: The torch graph to add the node to.
        opname: The name of the op to add. E.g. "onnx::Add".
        inputs: The onnx inputs to the op.
        attributes: The onnx attributes to the op.
        n_outputs: The number of outputs the op has.

    Returns:
        The outputs of the created node.
    """
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in attributes.items() if v is not None}

    node = graph.create(opname, inputs, n_outputs)
    node = graph.insertNode(node)
    node_ouputs = tuple(node.outputs())

    assert len(node_ouputs) == n_outputs
    # Add all attributes
    for key, value in sorted(attributes.items()):
        _add_attribute_to_torchscript_node(node, key, value)

    return node_ouputs


class TorchScriptGraph:
    def __init__(self):
        self._graph = torch.Graph()
        # All the functions used, deduplicated by name
        self._function_store: Dict[str, onnxscript.OnnxFunction] = {}

    @property
    def torch_graph(self):
        return self._graph

    @beartype
    def add_input(self, input_name: str, input_value: torch.Tensor) -> TorchScriptTensor:
        # TODO: Take in a TorchScriptTensor?
        # TODO: Support dynamic shapes
        if input_value is None:
            # This input argument is None, which is mapped
            # to a NULL value in TorchScript type system.
            torch_value = self._graph.op("prim::Constant")  # type: ignore[attr-defined]
            torch_value.setType(torch._C.OptionalType.ofTensor())
            tensor_value = _wrap_torch_value_to_tensor(torch_value)
            return tensor_value
        torch_value = self._graph.addInput(input_name)
        torch_value.setType(torch._C.TensorType.create_from_tensor(input_value))
        tensor_value = _wrap_torch_value_to_tensor(torch_value)
        return tensor_value

    @beartype
    def register_outputs(self, outputs: Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]):
        unwrapped_outputs = _unwrap_tensors_to_torch_values(outputs)
        if isinstance(unwrapped_outputs, torch.Value):
            self._graph.registerOutput(unwrapped_outputs)
            return
        assert isinstance(unwrapped_outputs, Sequence)
        for ts_output in unwrapped_outputs:
            assert isinstance(
                ts_output, torch.Value
            ), f"ts_output must be a torch.Value, not {type(ts_output)}"
            self._graph.registerOutput(ts_output)
        return

    def _add_constant_to_graph(self, constant) -> torch.Value:
        if isinstance(constant, float):
            return _create_op_call_in_torch_graph(
                self._graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.float)),
            )[0]
        if isinstance(constant, int):
            return _create_op_call_in_torch_graph(
                self._graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.int64)),
            )[0]
        if constant is None:
            value = _create_op_call_in_torch_graph(
                self._graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            value.setType(torch.OptionalType.ofTensor())
            return value
        if isinstance(constant, (tuple, list)) and all(
            isinstance(val, int) for val in constant
        ):
            return _create_op_call_in_torch_graph(
                self._graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.int64)),
            )[0]
        if isinstance(constant, (tuple, list)) and all(
            isinstance(val, float) for val in constant
        ):
            return _create_op_call_in_torch_graph(
                self._graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.float)),
            )[0]

        raise TypeError(
            f"Constant input `{constant}` of type '{type(constant)}' is not supported"
        )

    @beartype
    def _add_torchscript_op_call(
        self,
        name: str,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Dict[str, ValidArgumentType],
        n_outputs: int,
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        unwrapped_inputs = _unwrap_tensors_to_torch_values(onnx_inputs)
        graph_inputs = []
        assert isinstance(unwrapped_inputs, Sequence)
        for input in unwrapped_inputs:
            if not isinstance(input, torch.Value):
                graph_inputs.append(self._add_constant_to_graph(input))
            else:
                graph_inputs.append(input)
        for value in onnx_attributes.values():
            assert not isinstance(value, TorchScriptTensor)
        result = _create_op_call_in_torch_graph(
            self._graph,
            name,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            n_outputs=n_outputs,
        )
        if len(result) <= 1:
            return TorchScriptTensor(result[0])
        return tuple(TorchScriptTensor(v) for v in result)

    @beartype
    def add_op_call(
        self,
        onnx_op_schema: onnx.defs.OpSchema,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Dict[str, ValidArgumentType],
    ):
        # Compute outputs from the onnx_op op schema

        # FIXME(justinchuby): Figure out how to get the number of outputs from the schema
        result = self._add_torchscript_op_call(
            f"onnx::{onnx_op_schema.name}", onnx_inputs, onnx_attributes, n_outputs=1
        )

        return result

    @beartype
    def add_function_call(
        self,
        onnx_function: onnxscript.OnnxFunction,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Dict[str, ValidArgumentType],
    ):
        self._function_store[onnx_function.name] = onnx_function

        # Compute outputs from the function schema

        result = self._add_torchscript_op_call(
            f"{onnx_function.function_ir.domain}::{onnx_function.name}",
            onnx_inputs,
            onnx_attributes,
            n_outputs=len(onnx_function.function_ir.outputs),
        )

        return result

    @beartype
    def to_model_proto(
        self, initializers: Dict[str, torch.Tensor], opset_version: Optional[int]
    ) -> onnx.ModelProto:
        proto, _, _, _ = self._graph._export_onnx(
            initializers=initializers,
            onnx_opset_version=opset_version,
            # TODO(justinchuby): Figure out how to get the dynamic axes from the inputs
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
        #print("===========ONNX model: \n", onnx_model)
        #onnx_model = onnx.shape_inference.infer_shapes(onnx_model, True, True)
        #print("===========ONNX model with inferred shapes: \n", onnx_model)
        #onnx.checker.check_model(onnx_model, full_check=True)
        print("[Success] ONNX model exported")
        return onnx_model

    def apply(self, graph_pass: Callable, *args, **kwargs) -> None:
        """Apply a graph pass to the graph."""
        graph_pass(self._graph, *args, **kwargs)
