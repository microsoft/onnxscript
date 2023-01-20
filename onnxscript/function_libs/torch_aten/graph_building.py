"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import collections
import typing
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.shape_inference
import torch
from torch.onnx import _type_utils

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

    def symbolic_value(self) -> torch.Value:
        """The symbolic Value in torch.Graph."""
        return self._value


def _split_args_kwargs_to_input_attr(
    onnx_func: onnxscript.OnnxFunction, args, kwargs
) -> tuple[list[ValidArgumentType], dict[str, ValidArgumentType]]:
    """Split the function args and kwargs to onnx inputs and attributes.

    Args:
        onnx_func: The onnx function.
        args: The positional arguments.
        kwargs: The keyword arguments.

    Returns:
        A tuple of (onnx_inputs, onnx_attributes).
    """
    function_ir = onnx_func.function_ir
    # FIXME(justinchuby): There can be attributes in the args too.
    # The first len(func_ir.inputs) arguments are onnx inputs
    onnx_inputs = args[: len(function_ir.inputs)]
    # The rest is onnx attributes
    attributes_in_args = args[len(function_ir.inputs) :]
    # Construct a dictionary of attributes with their names specified in the function
    # definition
    onnx_attributes = {}

    # (1) Some/All attributes are supplied as positional arguments
    attr_name_to_protos = collections.OrderedDict(
        (attr.name, attr) for attr in function_ir.attr_protos
    )

    assert len(function_ir.attr_protos) >= len(attributes_in_args)
    for attr_proto, attr_value in zip(attr_name_to_protos.values(), attributes_in_args):
        onnx_attributes[attr_proto.name] = attr_value

    # (2) Some/All attributes are supplied as kwargs
    for key, value in kwargs.items():
        # (3) Some arguments in kwargs are not defined in the onnx function
        if key not in attr_name_to_protos:
            warnings.warn(f"Attribute '{key}' is not defined in the function definition")
            continue

        onnx_attributes[key] = value

    # (4) Fill in the default values from the attr_proto if not supplied by caller
    for key, attr_proto in attr_name_to_protos.items():
        if key not in onnx_attributes:
            onnx_attributes[key] = attr_proto.value

    return onnx_inputs, onnx_attributes


def _unwrap_tensor_to_torch_value(
    value: ValidArgumentType
    | dict[str, ValidArgumentType]
    | list[ValidArgumentType]
    | tuple[ValidArgumentType, ...]
) -> torch.Value | str | int | float | dict[str, torch.Value | str | int | float] | list[
    torch.Value | str | int | float
] | tuple[torch.Value | str | int | float, ...]:
    """Unwrap the TorchScriptTensor to torch.Value.

    Args:
        value: The value to unwrap.

    Returns:
        The unwrapped value.
    """
    if isinstance(value, dict):
        return {
            k: v.symbolic_value() if isinstance(v, TorchScriptTensor) else v
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [v.symbolic_value() if isinstance(v, TorchScriptTensor) else v for v in value]
    if isinstance(value, tuple):
        return tuple(
            v.symbolic_value() if isinstance(v, TorchScriptTensor) else v for v in value
        )
    if isinstance(value, TorchScriptTensor):
        return value.symbolic_value()
    # A normal python value
    return value


def _wrap_torch_value_to_tensor(
    value: torch.Value
    | dict[str, torch.Value | str | int | float]
    | list[torch.Value | str | int | float]
    | tuple[torch.Value | str | int | float, ...]
) -> ValidArgumentType | dict[str, ValidArgumentType] | list[ValidArgumentType] | tuple[
    ValidArgumentType, ...
]:
    """Wrap torch.Value to TorchScriptTensor.

    Args:
        value: The value to wrap.

    Returns:
        The wrapped value.
    """
    if isinstance(value, dict):
        return {
            k: TorchScriptTensor(v) if isinstance(v, torch.Value) else v
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [TorchScriptTensor(v) if isinstance(v, torch.Value) else v for v in value]
    if isinstance(value, tuple):
        return tuple(TorchScriptTensor(v) if isinstance(v, torch.Value) else v for v in value)
    if isinstance(value, torch.Value):
        return TorchScriptTensor(value)
    return value


def _unwrap_tensors_to_torch_values(tensors):
    # TODO(justinchuby): Do we really need this?
    if isinstance(tensors, Sequence):
        return [_unwrap_tensor_to_torch_value(output) for output in tensors]
    return _unwrap_tensor_to_torch_value(tensors)


class TorchScriptEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, graph: TorchScriptGraph):
        self._graph: TorchScriptGraph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    def eval_function(
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Sequence[ValidArgumentType],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        inputs, attributes = _split_args_kwargs_to_input_attr(function, args, kwargs)
        return self._graph.add_function(function, inputs, attributes)

    def _eval(self, schema: onnx.defs.OpSchema, inputs, attributes, closure: Any):
        # TODO(justinchuby): Does it really know what the inputs are?

        del closure  # Unused
        return self._graph.add_op(schema, inputs, attributes)


def _add_attribute_to_torchscrpt_node(
    node: torch.Node, key: str, value: float | int | str | Sequence[float] | Sequence[int]
):
    """Initializes the right attribute based on type of value."""
    if isinstance(value, float):
        kind = "f"
    elif isinstance(value, int):
        kind = "i"
    elif isinstance(value, str):
        kind = "s"
    elif isinstance(value, torch.Tensor):
        kind = "t"
    elif isinstance(value, Sequence):
        if isinstance(value, float):
            kind = "fs"
        elif isinstance(value, int):
            kind = "is"
        else:
            raise ValueError(
                f"Unsupported sequence type '{type(value)}' for attribute '{key}'"
            )
    else:
        raise ValueError(f"Unsupported attribute type '{type(value)}' for attribute '{key}'")
    return getattr(node, f"{kind}_")(key, value)


def _create_torchscript_op(
    graph: torch.Graph,
    opname: str,
    *,
    inputs: Sequence[torch.Value],
    attributes: dict[str, Any],
    n_outputs: int = 1,
) -> tuple[torch.Value, ...]:
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in attributes.items() if v is not None}

    node = graph.create(opname, inputs, n_outputs)
    node = graph.insertNode(node)
    node_ouputs = tuple(node.outputs())

    assert len(node_ouputs) == n_outputs
    # Add all attributes
    for key, value in sorted(attributes.items()):
        _add_attribute_to_torchscrpt_node(node, key, value)

    return tuple(node.outputs())


class TorchScriptGraph:
    def __init__(self):
        self._graph = torch.Graph()
        # All the functions used, deduplicated by name
        self._function_store: dict[str, onnxscript.OnnxFunction] = {}

    @property
    def graph(self):
        return self._graph

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
        assert isinstance(tensor_value, TorchScriptTensor)
        return tensor_value

    def register_outputs(
        self, outputs: Union[TorchScriptTensor, tuple[TorchScriptTensor, ...]]
    ):
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
            return _create_torchscript_op(
                self.graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.float)),
            )[0]
        if isinstance(constant, int):
            return _create_torchscript_op(
                self.graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.int64)),
            )[0]
        if constant is None:
            value = _create_torchscript_op(
                self.graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            value.setType(torch.OptionalType.ofTensor())
            return value
        if isinstance(constant, (tuple, list)) and all(
            isinstance(val, int) for val in constant
        ):
            return _create_torchscript_op(
                self.graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.int64)),
            )[0]
        if isinstance(constant, (tuple, list)) and all(
            isinstance(val, float) for val in constant
        ):
            return _create_torchscript_op(
                self.graph,
                "onnx::Constant",
                inputs=(),
                attributes=dict(value=torch.tensor(constant, dtype=torch.float)),
            )[0]

        raise TypeError(
            f"Constant input `{constant}` of type '{type(constant)}' is not supported"
        )

    def _add_torchscript_op(
        self,
        name,
        onnx_inputs,
        onnx_attributes,
        n_outputs: int,
    ) -> TorchScriptTensor | tuple[TorchScriptTensor, ...]:

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
        result = _create_torchscript_op(
            self.graph,
            name,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            n_outputs=n_outputs,
        )
        if len(result) <= 1:
            return TorchScriptTensor(result[0])
        return tuple(TorchScriptTensor(v) for v in result)

    def add_op(
        self,
        onnx_op_schema,
        onnx_inputs: Sequence[ValidArgumentType | Sequence[ValidArgumentType]],
        onnx_attributes: dict[str, ValidArgumentType | Sequence[ValidArgumentType]],
    ):
        # Compute outputs from the onnx_op op schema

        # FIXME(justinchuby): Figure out how to get the number of outputs from the schema
        result = self._add_torchscript_op(
            f"onnx::{onnx_op_schema.name}", onnx_inputs, onnx_attributes, n_outputs=1
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
            f"{onnx_function.function_ir.domain}::{onnx_function.name}",
            onnx_inputs,
            onnx_attributes,
            n_outputs=len(onnx_function.function_ir.outputs),
        )

        return result

    def to_model_proto(
        self, initializers: dict[str, torch.Tensor], opset_version: Optional[int]
    ) -> onnx.ModelProto:
        proto, _, _, _ = self.graph._export_onnx(
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
        print("===========ONNX model: \n", onnx_model)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        print("===========ONNX model with inferred shapes: \n", onnx_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        print("[Success] ONNX model exported")
        return onnx_model
