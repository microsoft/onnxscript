"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import logging
import os
import tempfile
import typing
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.helper
import onnx.shape_inference
import torch
from typing_extensions import TypeAlias

import onnxscript
from onnxscript import evaluator
from onnxscript import tensor as onnxscript_tensor
from onnxscript._internal import param_manipulation, runtime_typing
from onnxscript.function_libs.torch_lib import _flags
from onnxscript.function_libs.torch_lib.ops import common as common_ops

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

# Be sure to leave ample room for the rest of the proto fields.
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)  # 1.8GB

# TODO(justinchuby): Build a context manager to handle source information.


def _rename_intermediate_value(name: str) -> str:
    """Prepend `_val_` to a numeric tensor name make it valid in ONNX.

    The TorchScript graph creates numeric value names by default. e.g. `0`, `1`.
    These are not legal ONNX tensor names, since ONNX requires the names to be valid
    C variable names.

    It also improves readability by making the names less likely to be confused
    with shape values.
    """
    if name.isdigit():
        # Prefix with `_` to avoid name collision
        return f"_val_{name}"
    return name


class TorchScriptTensor(onnxscript_tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""

    def __init__(self, value: torch.Value):
        super().__init__(None)
        self._torch_value: torch.Value = value
        self._concrete_value: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int | None, ...]] = None
        self._torch_dtype: Optional[torch.dtype] = None
        self._name: Optional[str] = None
        self._is_complex: bool = False

    def __repr__(self):
        return f"TorchScriptTensor('{self._torch_value!r}')"

    @property  # type: ignore[override]
    def value(self) -> Optional[np.ndarray]:
        return self._concrete_value

    @value.setter
    def value(self, value: np.ndarray):
        self._concrete_value = value

    @property
    @runtime_typing.checked
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return self._torch_value.debugName()

    @name.setter
    @runtime_typing.checked
    def name(self, name: str):
        self._name = name
        self._torch_value.setDebugName(name)

    @property  # type: ignore[override]
    def rank(self) -> int | None:
        value_type = self._torch_value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
        return value_type.dim()

    @property  # type: ignore[override]
    def shape(self) -> Tuple[int | None, ...] | None:
        if self._shape is not None:
            return self._shape

        value_type = self._torch_value.type()
        if value_type is None:
            return None
        value_type = typing.cast(torch.TensorType, value_type)
        if isinstance(value_type, torch.OptionalType):
            shape = value_type.getElementType().varyingSizes()  # type: ignore[attr-defined]
        else:
            shape = value_type.varyingSizes()
        if shape is None:
            return None
        return tuple(shape)

    @shape.setter
    def shape(self, shape: Tuple[int | None, ...]):
        self._shape = shape
        self._torch_value.setType(self._torch_value.type().with_sizes(list(shape)))

    @property  # type: ignore[override]
    def dtype(self) -> torch.dtype | None:
        # TODO: Return numpy dtype
        if self._torch_dtype is not None:
            return self._torch_dtype
        # Local import to avoid circular dependency
        from torch.onnx import _type_utils  # pylint: disable=import-outside-toplevel

        torch_dtype = _type_utils.JitScalarType.from_value(  # type: ignore[attr-defined]
            self._torch_value, default=_type_utils.JitScalarType.UNDEFINED
        )
        if torch_dtype == _type_utils.JitScalarType.UNDEFINED:
            return None
        self._torch_dtype = torch_dtype.dtype()
        return self._torch_dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._torch_dtype = dtype
        self._torch_value.setType(self._torch_value.type().with_dtype(dtype))

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @is_complex.setter
    def is_complex(self, is_complex: bool):
        self._is_complex = is_complex

    @property
    def onnx_dtype(self):
        # Local import to avoid circular dependency
        from torch.onnx import _type_utils  # pylint: disable=import-outside-toplevel

        return _type_utils.JitScalarType.from_value(  # type: ignore[attr-defined]
            self._torch_value, _type_utils.JitScalarType.UNDEFINED
        ).onnx_type()

    def symbolic_value(self) -> torch.Value:
        """The symbolic Value in torch.Graph."""
        return self._torch_value


@runtime_typing.checked
def _unwrap_tensor_to_torch_value(
    value: Union[
        ValidArgumentType, Mapping[str, ValidArgumentType], Sequence[ValidArgumentType]
    ]
) -> Union[
    ValidTorchValueType,
    Dict[str, ValidTorchValueType],
    List[ValidTorchValueType],
    Tuple[ValidTorchValueType, ...],
]:
    """Unwrap the TorchScriptTensor to torch.Value."""
    if isinstance(value, TorchScriptTensor):
        return value.symbolic_value()
    if isinstance(value, dict):
        return {k: _unwrap_tensor_to_torch_value(v) for k, v in value.items()}  # type: ignore[misc,return-value]
    if isinstance(value, list):
        return [_unwrap_tensor_to_torch_value(v) for v in value]  # type: ignore[misc,return-value]
    if isinstance(value, tuple):
        return tuple(_unwrap_tensor_to_torch_value(v) for v in value)  # type: ignore[misc,return-value]

    # A normal python value
    return value  # type: ignore[return-value]


@runtime_typing.checked
def _wrap_torch_value_to_tensor(
    value: Union[torch.Value, Mapping[str, ValidTorchValueType], Sequence[ValidTorchValueType]]
) -> Union[
    ValidArgumentType,
    Dict[str, ValidArgumentType],
    List[ValidArgumentType],
    Tuple[ValidArgumentType, ...],
]:
    """Wrap torch.Value to TorchScriptTensor."""
    if isinstance(value, torch.Value):
        return TorchScriptTensor(value)
    if isinstance(value, dict):
        return {k: _wrap_torch_value_to_tensor(v) for k, v in value.items()}  # type: ignore[misc,return-value]
    if isinstance(value, list):
        return [_wrap_torch_value_to_tensor(v) for v in value]  # type: ignore[misc,return-value]
    if isinstance(value, tuple):
        return tuple(_wrap_torch_value_to_tensor(v) for v in value)  # type: ignore[misc,return-value]

    return value  # type: ignore[return-value]


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

    def eval(self, schema, inputs, attributes):
        return self._graph.add_op_call(schema, inputs, attributes)

    @runtime_typing.checked
    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Mapping[str, ValidArgumentType],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )
        name_to_schema = {param.name: param for param in param_schemas}
        for name, value in attributes.items():
            param = name_to_schema[name]
            # Cast int to float if needed
            if param.type in {float, "float"}:
                # FIXME(justinchuby): Create invariant on the type of param.type to simplify this
                attributes[name] = float(value)
        return self._graph.add_function_call(function, inputs, attributes)


@runtime_typing.checked
def _add_attribute_to_torchscript_node(
    node: torch.Node,
    key: str,
    value: Union[float, int, str, bytes, Sequence[float], Sequence[int], torch.Tensor],
):
    """Initializes the right attribute based on type of value."""
    if isinstance(value, float):
        return node.f_(key, value)
    if isinstance(value, int):
        return node.i_(key, value)
    if isinstance(value, (str, bytes)):
        return node.s_(key, value)  # type: ignore[arg-type]
    if isinstance(value, torch.Tensor):
        return node.t_(key, value)
    if isinstance(value, Sequence):
        if not value:
            # Treat empty sequences as empty list tensors
            # TODO(justinchuby): Revisit ways to determine the type of the empty list
            return node.is_(key, list(value))  # type: ignore[attr-defined]
        if isinstance(value[0], float):
            return node.fs_(key, list(value))  # type: ignore[arg-type]
        if isinstance(value[0], int):
            return node.is_(key, list(value))  # type: ignore[attr-defined]
        raise TypeError(f"Unsupported sequence type '{type(value)}' for attribute '{key}'")
    raise TypeError(f"Unsupported attribute type '{type(value)}' for attribute '{key}'")


@runtime_typing.checked
def _create_op_call_in_torch_graph(
    graph: torch.Graph,
    opname: str,
    *,
    inputs: Sequence[torch.Value],
    attributes: Mapping[str, Any],
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


def _tensor_rawdata_size(tensor: torch.Tensor) -> int:
    """Estimate the size of a tensor in bytes.

    Args:
        tensor: The tensor to estimate the size of.

    Returns:
        The estimated size of the tensor in bytes.
    """
    return tensor.numel() * tensor.element_size()


def _shared_functions() -> list[onnx.FunctionProto]:
    """Hack to always include the share ops."""

    # TODO: Remove after https://github.com/microsoft/onnxscript/issues/834 is fixed
    return [
        common_ops.Rank.to_function_proto(),
        common_ops.IsScalar.to_function_proto(),
    ]


class TorchScriptGraph:
    def __init__(
        self,
        parent_torch_script_graph: Optional[TorchScriptGraph] = None,
        domain_name: Optional[str] = None,
    ):
        self._torch_graph = torch.Graph()
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
        # Mapping from intializer name to data(torch.Tensor).
        self._initializers: Dict[str, torch.Tensor] = {}
        # Mapping from intializer name to input(TorchScriptTensor).
        self._initializers_inputs: Dict[str, TorchScriptTensor] = {}
        # Mapping from intializer name to input(TorchScriptTensor) from parent graph.
        self._initializers_inputs_from_parent: Dict[str, TorchScriptTensor] = {}
        # Mapping from model local function type name to function graph.
        # Local function type name is expected to be unique. Converter creates
        # a unique name and a unique function graph for every module call.
        self._sub_torch_script_graphs: Dict[str, TorchScriptGraph] = {}
        # Parent graph. None if this is the top level graph.
        self._parent_torch_script_graph = parent_torch_script_graph
        # Domain name of the graph. None if this is the top level graph.
        self._domain_name: Optional[str] = domain_name

        if self._domain_name is None and self._parent_torch_script_graph is not None:
            raise RuntimeError(
                "Domain name is not set. It is required because this 'TorchScriptGraph' instance "
                "is a subgraph that represents an ONNX local function."
            )

    @property
    def torch_graph(self):
        return self._torch_graph

    @property
    def initializers(self) -> Mapping[str, torch.Tensor]:
        return self._initializers

    # NOTE: This setter is used in torch converter when we activate fake mode,
    #       we need to filter out the initializers that has fake tensor. This
    #       is because we don't want to introduce fake tensor in onnxscript.
    @initializers.setter
    def initializers(self, initializers: Dict[str, torch.Tensor]):
        self._initializers = initializers

    @property
    def initializers_inputs(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs

    @property
    def initializers_inputs_from_parent(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs_from_parent

    @property
    def num_outputs(self) -> int:
        return len(list(self._torch_graph.outputs()))

    @property
    def domain_name(self) -> Optional[str]:
        return self._domain_name

    @runtime_typing.checked
    def add_input(
        self,
        input_name: Optional[str],
        shape: Optional[Union[torch.Size, Sequence[Union[int, str, None]]]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TorchScriptTensor:
        if input_name is None:
            # This input argument is None, which is mapped
            # to a NULL value in TorchScript type system.
            torch_value = _create_op_call_in_torch_graph(
                self._torch_graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            torch_value.setType(torch.OptionalType.ofTensor())
        else:
            torch_value = self._torch_graph.addInput(input_name)
            torch_value.setType(torch_value.type().with_dtype(dtype))  # type: ignore[arg-type]
            # TODO(titaiwang): This approach loses the information that "same SymInts
            # indicates same shape", for example, [symint0, symint0, symint1]
            # would all be [None, None, None]
            torch_value.setType(
                torch_value.type().with_sizes(
                    [dim if isinstance(dim, int) else None for dim in shape]  # type: ignore[union-attr]
                )
            )
        tensor_value = _wrap_torch_value_to_tensor(torch_value)
        return tensor_value  # type: ignore[return-value]

    @runtime_typing.checked
    def add_initializer(self, name: str, value: torch.Tensor) -> TorchScriptTensor:
        if name in self._initializers_inputs:
            # NOTE: Previously it raises when `name` is already set. This is relaxed
            # because this will be invoked multiple times when submodule is called
            # multiple times.
            if name in self._initializers and self._initializers[name] is not value:
                raise ValueError(
                    f"Initializer '{name}' exists already with a different value."
                )
            return self._initializers_inputs[name]  # type: ignore[return-value]

        if (
            self != self._parent_torch_script_graph
            and self._parent_torch_script_graph is not None
        ):
            # Only the root graph can have initializers. Add as initializer
            # to root graph, and add as input to current graph.
            self._initializers_inputs_from_parent[
                name
            ] = self._parent_torch_script_graph.add_initializer(name, value)
            torch_value = self._torch_graph.addInput(name)
            torch_value.setType(torch.TensorType.create_from_tensor(value))
            tensor_value = _wrap_torch_value_to_tensor(torch_value)
            self._initializers_inputs[name] = tensor_value  # type: ignore[assignment]
            return tensor_value  # type: ignore[return-value]

        self._initializers[name] = value
        torch_value = self._torch_graph.addInput(name)
        torch_value.setType(torch.TensorType.create_from_tensor(value))
        tensor_value = _wrap_torch_value_to_tensor(torch_value)
        self._initializers_inputs[name] = tensor_value  # type: ignore[assignment]
        return tensor_value  # type: ignore[return-value]

    @runtime_typing.checked
    def register_outputs(
        self, outputs: Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]
    ):
        unwrapped_outputs = _unwrap_tensors_to_torch_values(outputs)
        if isinstance(unwrapped_outputs, torch.Value):
            self._torch_graph.registerOutput(unwrapped_outputs)
            return
        assert isinstance(unwrapped_outputs, Sequence)
        for ts_output in unwrapped_outputs:
            assert isinstance(
                ts_output, torch.Value
            ), f"ts_output must be a torch.Value, not {type(ts_output)}"
            self._torch_graph.registerOutput(ts_output)
        return

    def _add_constant_to_graph(self, constant) -> torch.Value:
        if constant is None:
            value = _create_op_call_in_torch_graph(
                self._torch_graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            value.setType(torch.OptionalType.ofTensor())
            value.setDebugName(_rename_intermediate_value(value.debugName()))
            return value

        if isinstance(constant, bool):
            # Be sure to put bool before int, because bool is a subclass of int
            constant_tensor = torch.tensor(constant, dtype=torch.bool)
        elif isinstance(constant, float):
            constant_tensor = torch.tensor(constant, dtype=torch.float)
        elif isinstance(constant, int):
            constant_tensor = torch.tensor(constant, dtype=torch.int64)
        elif isinstance(constant, (tuple, list)) and all(
            isinstance(val, int) for val in constant
        ):
            constant_tensor = torch.tensor(constant, dtype=torch.int64)
        elif isinstance(constant, (tuple, list)) and all(
            isinstance(val, float) for val in constant
        ):
            constant_tensor = torch.tensor(constant, dtype=torch.float)
        else:
            raise TypeError(
                f"Constant input '{constant}' of type '{type(constant)}' is not supported"
            )
        value = _create_op_call_in_torch_graph(
            self._torch_graph,
            "onnx::Constant",
            inputs=(),
            attributes=dict(value=constant_tensor),
        )[0]
        value.setDebugName(_rename_intermediate_value(value.debugName()))
        return value

    @runtime_typing.checked
    def _add_torchscript_op_call(
        self,
        name: str,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
        n_outputs: int,
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        unwrapped_inputs = _unwrap_tensors_to_torch_values(onnx_inputs)
        graph_inputs = []
        assert isinstance(unwrapped_inputs, Sequence)
        for input in unwrapped_inputs:
            # NOTE(titaiwang): input could be empty list
            if (
                isinstance(input, Sequence)
                and input
                and all(isinstance(elem, torch.Value) for elem in input)
            ):
                # If all elements in the Sequence are torch.Values we know it
                # should be a Sequence input in ONNX.
                input_sequence = _create_op_call_in_torch_graph(
                    self._torch_graph,
                    "onnx::SequenceConstruct",
                    inputs=input,
                    attributes={},
                )[0]
                graph_inputs.append(input_sequence)
            elif not isinstance(input, torch.Value):
                graph_inputs.append(self._add_constant_to_graph(input))
            else:
                graph_inputs.append(input)
        for key, value in onnx_attributes.items():
            assert not isinstance(
                value, TorchScriptTensor
            ), f"ONNX attribute must not be a TorchScriptTensor, got {key}: {value}."
        result = _create_op_call_in_torch_graph(
            self._torch_graph,
            name,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            n_outputs=n_outputs,
        )
        assert result, "Expected at least one output from ONNX op call."
        if len(result) == 1:
            tensor = TorchScriptTensor(result[0])
            tensor.name = _rename_intermediate_value(tensor.name)
            return tensor
        tensors = tuple(TorchScriptTensor(v) for v in result)
        for tensor in tensors:
            tensor.name = _rename_intermediate_value(tensor.name)
        return tensors

    @runtime_typing.checked
    def fetch_function_proto_dict(
        self, opset_version: int
    ) -> Mapping[Tuple[str, str], onnx.FunctionProto]:
        function_proto_dict: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        # Fetch local function protos. E.g., local functions representing module calls.
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_proto_dict.update(
                sub_torch_script_graph.fetch_function_proto_dict(opset_version)
            )
            domain = sub_torch_script_graph.domain_name
            assert domain is not None
            name_domain = (
                sub_graph_name,
                domain,
            )
            assert (
                name_domain not in function_proto_dict
            ), f"Sub graph name already exists. {name_domain}"
            function_proto_dict[name_domain] = sub_torch_script_graph.to_function_proto(
                opset_version, sub_graph_name
            )
        # Fetch torchlib function protos.
        for name_domain, function in self._function_store.items():
            function_proto_dict[name_domain] = function.to_function_proto()
        return function_proto_dict

    @runtime_typing.checked
    def add_op_call(
        self,
        onnx_op_schema: onnx.defs.OpSchema,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        # Compute outputs from the onnx_op op schema
        n_outputs = evaluator.compute_num_outputs(onnx_op_schema, onnx_inputs, onnx_attributes)
        result = self._add_torchscript_op_call(
            f"onnx::{onnx_op_schema.name}",
            onnx_inputs,
            onnx_attributes,
            n_outputs=n_outputs,
        )

        return result

    @runtime_typing.checked
    def add_function_call(
        self,
        onnx_function: onnxscript.OnnxFunction,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        identifier = (onnx_function.name, onnx_function.function_ir.domain)
        self._function_store[identifier] = onnx_function

        # Compute outputs from the function schema
        result = self._add_torchscript_op_call(
            f"{onnx_function.function_ir.domain}::{onnx_function.name}",
            onnx_inputs,
            onnx_attributes,
            n_outputs=len(onnx_function.function_ir.outputs),
        )

        return result

    @runtime_typing.checked
    def add_module_call(
        self,
        name: str,
        sub_torch_script_graph: TorchScriptGraph,
        onnx_inputs: Sequence[ValidInputType],
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        self._sub_torch_script_graphs[name] = sub_torch_script_graph
        domain_name = sub_torch_script_graph.domain_name
        assert domain_name is not None
        return self._add_torchscript_op_call(
            f"{domain_name}::{name}",
            onnx_inputs=(
                *onnx_inputs,
                *sub_torch_script_graph.initializers_inputs_from_parent.values(),
            ),
            onnx_attributes={},
            n_outputs=sub_torch_script_graph.num_outputs,
        )

    @runtime_typing.checked
    def to_function_proto(self, opset_version: int, function_name: str) -> onnx.FunctionProto:
        assert len(self.initializers) == 0, "Model local functions cannot have initializers."
        (
            proto,
            _,
            _,
            _,
        ) = self._torch_graph._export_onnx(  # type: ignore[attr-defined] # pylint: disable=protected-access
            initializers={},
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=False,
            custom_opsets={},
            add_node_names=True,
            onnx_file_path="",  # Large model export. Out of scope.
            node_attr_to_name={},  # Current module as function feature does not utilize attributes.
        )

        onnx_model = onnx.load_from_string(proto)

        # Dissect the model proto and transform to function proto.
        domain = self.domain_name
        if domain is None:
            raise RuntimeError("Domain name is not set.")
        onnx_function = onnx.helper.make_function(
            domain=domain,
            fname=function_name,
            inputs=[input.name for input in onnx_model.graph.input],
            outputs=[output.name for output in onnx_model.graph.output],
            nodes=onnx_model.graph.node,
            opset_imports=onnx_model.opset_import,
            doc_string=onnx_model.doc_string,
        )
        return onnx_function

    @runtime_typing.checked
    def to_model_proto(
        self, opset_version: int, include_initializers: bool = True
    ) -> onnx.ModelProto:
        function_proto_dict: Mapping[
            Tuple[str, str], onnx.FunctionProto
        ] = self.fetch_function_proto_dict(opset_version)
        unique_custom_domains: Dict[str, int] = {}

        for function_proto in function_proto_dict.values():
            # TODO(BowenBao): All local function domain versions are hardcoded as 1.
            unique_custom_domains[function_proto.domain] = 1

        initializers_size = sum(
            _tensor_rawdata_size(tensor) for tensor in self.initializers.values()
        )

        large_model = initializers_size > _LARGE_MODEL_SIZE_THRESHOLD

        export_kwargs: dict[str, Any] = dict(
            initializers=self.initializers
            if include_initializers and not _flags.EXPERIMENTAL_INITIALIZERS_AS_INPUTS
            else {},
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=_flags.EXPERIMENTAL_INITIALIZERS_AS_INPUTS,
            custom_opsets={},
            add_node_names=True,
            node_attr_to_name={},
        )

        # We decided to cache the model to disk when the model is large.
        # Alternatively, we could build the ONNX `TensorProto`s in memory
        # and append them to the model proto.
        # We did not do it because it is harder to get right (vs. PyTorch's battle-tested
        # implementation) and creating the `TensorProto`s naively (by converting to numpy)
        # is slow.
        cache_model_to_disk = large_model and include_initializers

        if cache_model_to_disk:
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_file_path = os.path.join(temp_dir, "exported_model.onnx")
                export_kwargs["onnx_file_path"] = onnx_file_path
                (
                    proto,
                    _,
                    _,
                    _,
                ) = self._torch_graph._export_onnx(  # type: ignore[attr-defined] # pylint: disable=protected-access
                    **export_kwargs
                )
                onnx_model = onnx.load_from_string(proto)
                onnx.load_external_data_for_model(onnx_model, temp_dir)
        else:
            (
                proto,
                _,
                _,
                _,
            ) = self._torch_graph._export_onnx(  # type: ignore[attr-defined] # pylint: disable=protected-access
                **export_kwargs
            )
            onnx_model = onnx.load_from_string(proto)

        onnx_model.functions.extend(function_proto_dict.values())
        onnx_model.functions.extend(_shared_functions())

        # `_export_onnx` only exports opset_imports that is visible to it. It does not
        # export opset_imports for nested functions, since it does not have access to
        # them. We manually add them back and merge with existing opset_imports in the
        # model proto.
        while len(onnx_model.opset_import) > 0:
            opsetid = onnx_model.opset_import.pop()
            unique_custom_domains[opsetid.domain] = opsetid.version
        onnx_model.opset_import.extend(
            [
                onnx.helper.make_opsetid(domain, version)
                for domain, version in unique_custom_domains.items()
            ]
        )
        # Include the library shared opset domain
        # TODO: Remove after https://github.com/microsoft/onnxscript/issues/834 is fixed
        onnx_model.opset_import.append(
            onnx.helper.make_opsetid(
                common_ops.common_opset.domain, common_ops.common_opset.version
            )
        )

        try:
            if not cache_model_to_disk:
                # Only check the model if it is in memory.
                # Otherwise the checker and shape_inference will fail because
                # we cannot serialize the model.
                onnx_model = onnx.shape_inference.infer_shapes(
                    onnx_model, check_type=True, strict_mode=False, data_prop=True
                )
                onnx.checker.check_model(onnx_model, full_check=True)
        except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as e:
            warnings.warn(f"ONNX model is invalid: {e}", stacklevel=1)
            logging.debug(
                "ONNX model:\n%s\n\nTorchScript graph:\n%s",
                onnxscript.proto2text(onnx_model),
                self.torch_graph,
            )
        return onnx_model
