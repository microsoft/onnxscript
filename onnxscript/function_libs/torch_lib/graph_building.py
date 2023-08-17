"""Graph building functions for torchscript graph backend."""
from __future__ import annotations

import logging
import os
import tempfile
import typing
import warnings
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.helper
import onnx.shape_inference
import torch
from torch.onnx import _type_utils
from typing_extensions import TypeAlias
import spox
from spox import _node
import spox.opset.ai.onnx.v18 as spox_op

import onnxscript
from onnxscript import evaluator
from onnxscript import tensor as onnxscript_tensor
from onnxscript._internal import param_manipulation, runtime_typing

__all__ = [
    "SymbolicTensor",
    "TorchScriptGraph",
    "TorchScriptTracingEvaluator",
]


ValidArgumentType: TypeAlias = Union[
    "SymbolicTensor",
    Sequence["SymbolicTensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidInputType: TypeAlias = Union[
    "SymbolicTensor",
    Sequence["SymbolicTensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidSpoxValueType: TypeAlias = Union[
    spox.Var,
    Sequence[spox.Var],
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

TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128
}

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


class SymbolicTensor(onnxscript_tensor.Tensor):
    """A symbolic tensor."""

    def __init__(self, value: spox.Var):
        super().__init__(None)
        self._spox_value: spox.Var = value
        self._concrete_value: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int | str | None, ...]] = None
        self._torch_dtype: Optional[torch.dtype] = None
        self._name: Optional[str] = None
        self._is_complex: bool = False

    def __repr__(self):
        return f"SymbolicTensor('{self._spox_value!r}')"

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
        return self._spox_value._name

    @name.setter
    @runtime_typing.checked
    def name(self, name: str):
        self._name = name
        self._spox_value._rename(name)

    @property  # type: ignore[override]
    def rank(self) -> int | None:
        return len(self.shape)

    @property  # type: ignore[override]
    def shape(self) -> Tuple[int | str | None, ...] | None:
        shape = self._spox_value.unwrap_tensor().shape
        assert shape
        return shape

    @shape.setter
    def shape(self, shape: Tuple[int | str | None, ...]):
        self._shape = shape
        self._spox_value.type = spox.Tensor(self._spox_value.unwrap_tensor().dtype, self.shape)


    @property  # type: ignore[override]
    def dtype(self) -> torch.dtype | None:
        # TODO: Return numpy dtype
        if self._torch_dtype is not None:
            return self._torch_dtype

        raise NotImplementedError()

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._torch_dtype = dtype
        np_dtype = TORCH_TO_NUMPY_DTYPE[dtype]
        # TODO(justinchuby): Handle when shape is not set
        self._spox_value.type = spox.Tensor(np_dtype, self.shape)

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @is_complex.setter
    def is_complex(self, is_complex: bool):
        self._is_complex = is_complex

    def symbolic_value(self) -> spox.Var:
        """The symbolic Value in torch.Graph."""
        return self._spox_value


@runtime_typing.checked
def _unwrap_tensor_to_spox_value(
    value: Union[
        ValidArgumentType, Mapping[str, ValidArgumentType], Sequence[ValidArgumentType]
    ]
) -> Union[
    ValidSpoxValueType,
    Dict[str, ValidSpoxValueType],
    List[ValidSpoxValueType],
    Tuple[ValidSpoxValueType, ...],
]:
    """Unwrap the SymbolicTensor to torch.Value."""
    if isinstance(value, SymbolicTensor):
        return value.symbolic_value()
    if isinstance(value, dict):
        return {k: _unwrap_tensor_to_spox_value(v) for k, v in value.items()}  # type: ignore[misc,return-value]
    if isinstance(value, list):
        return [_unwrap_tensor_to_spox_value(v) for v in value]  # type: ignore[misc,return-value]
    if isinstance(value, tuple):
        return tuple(_unwrap_tensor_to_spox_value(v) for v in value)  # type: ignore[misc,return-value]

    # A normal python value
    return value  # type: ignore[return-value]


@runtime_typing.checked
def _wrap_spox_value_to_tensor(
    value: Union[spox.Var, Mapping[str, ValidSpoxValueType], Sequence[ValidSpoxValueType]]
) -> Union[
    ValidArgumentType,
    Dict[str, ValidArgumentType],
    List[ValidArgumentType],
    Tuple[ValidArgumentType, ...],
]:
    """Wrap spox.Var to SymbolicTensor."""
    if isinstance(value, spox.Var):
        return SymbolicTensor(value)
    if isinstance(value, dict):
        return {k: _wrap_spox_value_to_tensor(v) for k, v in value.items()}  # type: ignore[misc,return-value]
    # TODO(justinchuby): Should we wrap the list into a Sequence and treat it
    # differently?
    if isinstance(value, list):
        return [_wrap_spox_value_to_tensor(v) for v in value]  # type: ignore[misc,return-value]
    if isinstance(value, tuple):
        return tuple(_wrap_spox_value_to_tensor(v) for v in value)  # type: ignore[misc,return-value]

    return value  # type: ignore[return-value]


def _unwrap_tensors_to_spox_values(tensors):
    # TODO(justinchuby): Do we really need this?
    if isinstance(tensors, Sequence):
        return [_unwrap_tensor_to_spox_value(output) for output in tensors]
    return _unwrap_tensor_to_spox_value(tensors)


class SpoxTracingEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into Spox."""

    def __init__(self, graph: SpoxGraph):
        self._graph: SpoxGraph = graph

    @property
    def graph(self) -> SpoxGraph:
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
        # args/kwargs are SymbolicTensor/python built-in based
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
def _add_attribute_to_spox_node(
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
    graph: spox.Graph,
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
        _add_attribute_to_spox_node(node, key, value)

    return node_ouputs


def _tensor_rawdata_size(tensor: torch.Tensor) -> int:
    """Estimate the size of a tensor in bytes.

    Args:
        tensor: The tensor to estimate the size of.

    Returns:
        The estimated size of the tensor in bytes.
    """
    return tensor.numel() * tensor.element_size()



class SpoxGraph:
    _LOCAL_FUNCTION_DOMAIN_NAME: Final[str] = "torch_export"
    """The domain name for local functions."""

    def __init__(self, parent_torch_script_graph: Optional[SpoxGraph] = None):
        # All nodes inside this graph
        self._nodes: List[_node.Node] = {}
        # The outputs of this graph
        self._outputs: List[SymbolicTensor] = []
        # The inputs of the graph
        self._inputs: List[SymbolicTensor] = []
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
        # Mapping from intializer name to data(torch.Tensor).
        self._initializers: Dict[str, torch.Tensor] = {}
        # Mapping from intializer name to input(SymbolicTensor).
        self._initializers_inputs: Dict[str, SymbolicTensor] = {}
        # Mapping from intializer name to input(SymbolicTensor) from parent graph.
        self._initializers_inputs_from_parent: Dict[str, SymbolicTensor] = {}
        # Mapping from model local function type name to function graph.
        # Local function type name is expected to be unique. Converter creates
        # a unique name and a unique function graph for every module call.
        self._sub_torch_script_graphs: Dict[str, TorchScriptGraph] = {}
        # Parent graph. None if this is the top level graph.
        self._parent_torch_script_graph = parent_torch_script_graph

    @property
    def initializers(self) -> Mapping[str, torch.Tensor]:
        return self._initializers

    @property
    def initializers_inputs(self) -> Mapping[str, SymbolicTensor]:
        return self._initializers_inputs

    @property
    def initializers_inputs_from_parent(self) -> Mapping[str, SymbolicTensor]:
        return self._initializers_inputs_from_parent

    @property
    def num_outputs(self) -> int:
        return len(self._outputs)

    @runtime_typing.checked
    def add_input(
        self,
        input_name: Optional[str],
        shape: Optional[Union[torch.Size, Sequence[Union[int, str, None]]]],
        dtype: Optional[torch.dtype],
    ) -> SymbolicTensor:
        # Create a spox variable and register the name as an input
        # TODO(justinchuby): converter the torch dtype to numpy dtype
        input = spox.argument(spox.Tensor(dtype, shape))
        if input_name is not None:
            input._rename(input_name)
        # that is not an input to the graph?
        tensor_value = _wrap_spox_value_to_tensor(input)
        self._inputs.append(tensor_value)
        return tensor_value  # type: ignore[return-value]

    @runtime_typing.checked
    def add_initializer(self, name: str, value: torch.Tensor) -> SymbolicTensor:
        # TODO(justinchuby): How do we record the initializer

        # 1. Add the input to the graph
        # 2. Record the initializer to the store
        # 3. Return the SymbolicTensor
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
        else:
            self._initializers[name] = value

        symbolic_tensor = self.add_input(name, value.shape, value.dtype)
        self._initializers_inputs[name] = symbolic_tensor
        return symbolic_tensor

    @runtime_typing.checked
    def register_outputs(
        self, outputs: Union[SymbolicTensor, Tuple[SymbolicTensor, ...]]
    ) -> None:
        # TODO(justinchuby): We dont need to unwrap the tensors here, because
        # we manage the graph ourselves
        assert isinstance(outputs, SymbolicTensor) or isinstance(
            outputs, tuple
        ), f"outputs must be a SymbolicTensor or Sequence, not {type(outputs)}"
        if isinstance(outputs, tuple):
            self._outputs.extend(outputs)
            return
        self._outputs.append(outputs)

    def _add_constant_to_graph(self, constant) -> spox.Var:
        # TODO(justinchuby): Figure out how the add constants in spox
        # if constant is None:
        #     value = _create_op_call_in_torch_graph(
        #         self._torch_graph, "prim::Constant", inputs=(), attributes={}
        #     )[0]
        #     value.setType(torch.OptionalType.ofTensor())
        #     value.setDebugName(_rename_intermediate_value(value.debugName()))
        #     return value

        if isinstance(constant, (bool, float, int)):
            constant_tensor = spox_op.constant(value=np.array(constant))
        elif isinstance(constant, (tuple, list)) and all(
            isinstance(val, (int, float)) for val in constant
        ):
            constant_tensor = spox_op.constant(value=np.array(constant))
        else:
            raise TypeError(
                f"Constant input '{constant}' of type '{type(constant)}' is not supported"
            )
        # TODO(justinchuby): Should we store the name of the node here or have
        # a handle to the node at all? How can we get the underlying node
        # from the tensor? Or should we create the node ourselves?
        # TODO(justinchuby): Cannot rename right now because the tensor is not
        # even in the graph yet?
        # value.setDebugName(_rename_intermediate_value(value.debugName()))
        # TODO(justinchuby): Should we return the raw value here or should we
        # return the SymbolicTensor?
        return constant_tensor

    @runtime_typing.checked
    def _create_op_call(
        self,
        operator_name: str,
        inputs: Sequence[spox.Var],
        attributes: Mapping[str, Any],
    ) -> Tuple[spox.Var]:
        # TODO(justinchuby): Create a node using the inputs
        # Then use the "outputs" of the node to return Vars.
        # The names of the output can be created dynamically.
        # We also need to dynamically create the BaseInputs and BaseOutputs
        # which are dataclasses.
        # TODO(justinchuby): The Node class is designed to be created declaratively
        # and is the same concept is an `Op` in onnxscript
        # TODO(justinchuby):
        # I don't like how we need to massage dataclasses for inputs and outputs
        # for it. Ideally we can just say here's the inputs and outputs and the
        # graph is created
        ...

    @runtime_typing.checked
    def _add_torchscript_op_call(
        self,
        name: str,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
        n_outputs: int,
    ) -> Union[SymbolicTensor, Tuple[SymbolicTensor, ...]]:
        # TODO(justinchuby)
        # 1. Unwrap the inputs
        # 2. Should we change how we represent attributes? Maybe not?
        # 3. Create the node, connect it, register it in the graph, and return the outputs
        unwrapped_inputs = _unwrap_tensors_to_spox_values(onnx_inputs)
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
                # TODO(justinchuby): Remember to handle the domain as well?
                input_sequence = self._create_op_call("SequenceConstruct", input, {})[0]
                graph_inputs.append(input_sequence)
            # TODO(justinchuby): MARK: Tired. I am here
            elif not isinstance(input, torch.Value):
                graph_inputs.append(self._add_constant_to_graph(input))
            else:
                graph_inputs.append(input)
        for key, value in onnx_attributes.items():
            assert not isinstance(
                value, SymbolicTensor
            ), f"ONNX attribute must not be a SymbolicTensor, got {key}: {value}."
        result = _create_op_call_in_torch_graph(
            self._torch_graph,
            name,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            n_outputs=n_outputs,
        )
        assert result, "Expected at least one output from ONNX op call."
        if len(result) == 1:
            tensor = SymbolicTensor(result[0])
            tensor.name = _rename_intermediate_value(tensor.name)
            return tensor
        tensors = tuple(SymbolicTensor(v) for v in result)
        for tensor in tensors:
            tensor.name = _rename_intermediate_value(tensor.name)
        return tensors

    @runtime_typing.checked
    def fetch_function_proto_dict(
        self, opset_version: int
    ) -> Mapping[Tuple[str, str], onnx.FunctionProto]:
        function_proto_dict: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_proto_dict.update(
                sub_torch_script_graph.fetch_function_proto_dict(opset_version)
            )
            name_domain = (
                sub_graph_name,
                self._LOCAL_FUNCTION_DOMAIN_NAME,
            )
            assert (
                name_domain not in function_proto_dict
            ), f"Sub graph name already exists. {name_domain}"
            function_proto_dict[name_domain] = sub_torch_script_graph.to_function_proto(
                opset_version, sub_graph_name
            )
        for name_domain, function in self._function_store.items():
            function_proto_dict[name_domain] = function.to_function_proto()
        return function_proto_dict

    @runtime_typing.checked
    def add_op_call(
        self,
        onnx_op_schema: onnx.defs.OpSchema,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
    ) -> Union[SymbolicTensor, Tuple[SymbolicTensor, ...]]:
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
    ) -> Union[SymbolicTensor, Tuple[SymbolicTensor, ...]]:
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
    ) -> Union[SymbolicTensor, Tuple[SymbolicTensor, ...]]:
        self._sub_torch_script_graphs[name] = sub_torch_script_graph
        return self._add_torchscript_op_call(
            f"{self._LOCAL_FUNCTION_DOMAIN_NAME}::{name}",
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
        onnx_function = onnx.helper.make_function(
            domain=self._LOCAL_FUNCTION_DOMAIN_NAME,
            fname=function_name,
            inputs=[input.name for input in onnx_model.graph.input],
            outputs=[output.name for output in onnx_model.graph.output],
            nodes=onnx_model.graph.node,
            opset_imports=onnx_model.opset_import,
            doc_string=onnx_model.doc_string,
        )
        # TODO: onnx.checker.check_function(onnx_function)?
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
            initializers=self.initializers if include_initializers else {},
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=False,
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
        cache_model_to_disk = include_initializers and large_model

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
