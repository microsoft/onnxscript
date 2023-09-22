# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.helper
import onnx.shape_inference
from typing_extensions import TypeAlias

import onnxscript
from onnxscript import evaluator, onnx_opset
from onnxscript import tensor as onnxscript_tensor
from onnxscript._internal import param_manipulation, runtime_typing

__all__ = [
    "Graph",
    "GraphEvaluator",
]


ValidArgumentType: TypeAlias = Union[
    "Tensor",
    Sequence["Tensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]


# TODO(titaiwang): Should we make onnxscript.Tensor a subclass of a
# more general Tensor class? In this Tensor, we expect it to have
# a name, shape, dtype, and value. The value is optional, and
# if it is not set, then it is a "fake" tensor.
class Tensor(onnxscript_tensor.Tensor):
    """An implementation of ONNX Tensors, based on a wrapper around numpy arrays.
    Serves to define overloaded ops with an ONNX/ONNXScript semantics.
    """

    def __init__(
        self,
        name: str = "",
        nparray: Optional[np.ndarray] = None,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        opset=None,
        onnx_type: str = "",
    ):
        super().__init__(nparray=nparray)
        if nparray is not None and not isinstance(nparray, np.ndarray):
            raise TypeError(
                f"Unexpected type {type(nparray)}. It must be a numpy array or None."
            )
        self._name = name
        self._nparray = nparray
        self._shape = nparray.shape if nparray is not None else shape
        self._dtype = nparray.dtype if nparray is not None else dtype
        # FIXME(justinhuby): Create a better way to determine the opset version
        self._opset: Any = opset or onnx_opset.opset18
        self._onnx_type = onnx_type

    # NOTE: Optional tensor
    @staticmethod
    def empty() -> Tensor:
        return Tensor(name="")

    def to_value_info(self) -> onnx.ValueInfoProto:
        # TODO: support more types?
        if self.onnx_type == "tensot_type":
            return onnx.helper.make_tensor_value_info(self.name, self.onnx_dtype, self.shape)  # type: ignore[arg-type]
        if self.onnx_type == "sequence_type":
            return onnx.helper.make_tensor_sequence_value_info(
                self.name, self.onnx_dtype, self.shape  # type: ignore[arg-type]
            )
        if self.onnx_type == "optional_type":
            # NOTE: no `make_optional_value_info` API
            optional_type_proto = onnx.helper.make_optional_type_proto(
                onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[self.dtype]  # type: ignore[arg-type]
            )
            return onnx.helper.make_value_info(self.name, optional_type_proto)
        return onnx.helper.make_empty_tensor_value_info(self.name)

    @property
    def onnx_type(self) -> str:
        return self._onnx_type

    @property
    def name(self) -> str:
        return self._name

    # TODO(titaiwang): better name?
    @property
    def is_fake(self) -> bool:
        return self._nparray is None

    @property
    def value(self) -> np.ndarray:
        if self._nparray is None:
            raise ValueError("Tensor does not have a value.")
        return self._nparray

    @property
    def shape(self) -> tuple[int, ...] | None:  # type: ignore[override]
        return self._shape

    @property
    def rank(self) -> int:
        return len(self.shape) if self.shape is not None else 0

    @property
    def dtype(self) -> np.dtype | None:  # type: ignore[override]
        return self._dtype

    @property
    def onnx_dtype(self) -> int | None:  # type: ignore[override]
        return (
            onnx.helper.np_dtype_to_tensor_dtype(self.dtype)
            if self.dtype is not None
            else None
        )


###################################################################################################


class OpNode:
    def __init__(
        self,
        namespace: str,
        op_name: str,
        inputs: List[Tensor],
        attributes: Dict[str, ValidArgumentType],
        n_outputs: int,
    ):
        self._namespace = namespace
        self._op_name = op_name
        self._inputs = inputs
        self._attributes = attributes
        self._n_outputs = n_outputs


class FunctionNode:
    def __init__(
        self,
        function: onnxscript.OnnxFunction,
        inputs: List[Tensor],
        attributes: Dict[str, ValidArgumentType],
        n_outputs: int,
    ):
        self._function = function
        self._inputs = inputs
        self._attributes = attributes
        self._n_outputs = n_outputs
        self.namespace = function.function_ir.domain
        self.op_name = function.name

    @property
    def function(self) -> onnxscript.OnnxFunction:
        return self._function


class ModuleNode:
    def __init__(self, subgraph: Graph, inputs: List[Tensor], n_outputs: int, name: str):
        self._subgraph = subgraph
        self._inputs = inputs
        self._n_outputs = n_outputs
        self.namespace = subgraph.domain_name
        self.op_name = name
        self._attributes: Dict[str, ValidArgumentType] = {}

    @property
    def module(self) -> Graph:
        return self._subgraph


class Node(abc.ABC):
    def __init__(self):
        self._inputs: List[Tensor] = []
        self._outputs: List[Tensor] = []
        self._attributes: Dict[str, ValidArgumentType] = {}
        self._domain: str = ""
        self._op_name: str = ""

    @property
    def inputs(self) -> List[Tensor]:
        return self._inputs

    @property
    def outputs(self) -> List[Tensor]:
        return self._outputs

    @property
    def attributes(self) -> Dict[str, ValidArgumentType]:
        return self._attributes

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def op_name(self) -> str:
        return self._op_name

    def to_node_proto(self) -> onnx.NodeProto:
        onnx_node = onnx.helper.make_node(
            self.op_name,
            inputs=[t.name for t in self.inputs],
            outputs=[t.name for t in self.outputs],
            domain=self.domain,
            **self.attributes,  # TODO: check if this works
        )
        return onnx_node

    @classmethod
    def from_op_schema(
        cls,
        op_schema: onnx.defs.OpSchema,
        inputs: List[Tensor],
        attributes: Dict[str, ValidArgumentType],
        n_outputs: int,
    ) -> OpNode:
        namespace = op_schema.domain
        op_name = op_schema.name
        return OpNode(
            namespace=namespace,
            op_name=op_name,
            inputs=inputs,
            attributes=attributes,
            n_outputs=n_outputs,
        )

    @classmethod
    def from_function(
        cls,
        onnx_function: onnxscript.OnnxFunction,
        inputs: List[Tensor],
        attributes: Dict[str, ValidArgumentType],
        n_outputs: int,
    ) -> FunctionNode:
        return FunctionNode(
            function=onnx_function,
            inputs=inputs,
            attributes=attributes,
            n_outputs=n_outputs,
        )

    @classmethod
    def from_module(cls, subgraph: Graph, name: str, inputs: List[Tensor]) -> ModuleNode:
        return ModuleNode(
            subgraph=subgraph,
            inputs=inputs,
            n_outputs=len(subgraph.outputs),
            name=name,
        )


# TODO(titaiwang): How we deal with subgraph?
class Graph:
    def __init__(
        self,
        producer_name="pytorch",
        parent_graph: Optional[Graph] = None,
        domain_name: Optional[str] = None,
    ) -> None:
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
        self._nodes: List[Node] = []
        self._inputs: List[Tensor] = []
        self._outputs: List[Tensor] = []
        self._initializers: Dict[str, Tensor] = {}
        self._producer_name: str = producer_name
        # TODO: what are these two string for?
        self._doc_string: str = ""
        self._name: str = ""
        # NOTE: below are used by splitting subgraphs
        # Mapping from intializer name to input(ensor).
        self._initializers_inputs: Dict[str, Tensor] = {}
        # Mapping from intializer name to input(TorchScriptTensor) from parent graph.
        self._initializers_inputs_from_parent: Dict[str, Tensor] = {}
        # Mapping from model local function type name to function graph.
        # Local function type name is expected to be unique. Converter creates
        # a unique name and a unique function graph for every module call.
        self._sub_torch_script_graphs: Dict[str, Graph] = {}
        # Parent graph. None if this is the top level graph.
        self._parent_torch_script_graph = parent_graph
        # Domain name of the graph. None if this is the top level graph.
        self._domain_name: Optional[str] = domain_name

        if self._domain_name is None and self._parent_torch_script_graph is not None:
            raise RuntimeError(
                "Domain name is not set. It is required because this 'TorchScriptGraph' instance "
                "is a subgraph that represents an ONNX local function."
            )

    @property
    def producer_name(self) -> str:
        return self._producer_name

    @property
    def outputs(self) -> List[Tensor]:
        return self._outputs

    @property
    def domain_name(self) -> Optional[str]:
        return self._domain_name

    @runtime_typing.checked
    def add_node(self, node: Node):
        if isinstance(node, FunctionNode):
            identifier = (node.op_name, node.namespace)
            self._function_store[identifier] = node.function
        elif isinstance(node, ModuleNode):
            self._sub_torch_script_graphs[node.op_name] = node.module
        self._nodes.append(node)

    # TODO(titaiwang): Should we not expose Tensor?
    # NOTE: We don't need `add_initializer` because we can just use `add_input`
    # for constant tensors
    @runtime_typing.checked
    def add_input(self, tensor: Tensor):
        self._inputs.append(tensor)

    @runtime_typing.checked
    def add_output(self, tensor: Tensor):
        self._outputs.append(tensor)

    # TODO(titaiwang): this function is not friendly to users when
    # they want to build the graph from scratch, as they need to
    # manually set the value to initializers, so we need another
    # method to get all initializers from the graph `self.initializers`
    @runtime_typing.checked
    def add_initializer(self, name: str, value: Tensor):
        if name in self._initializers_inputs:
            # NOTE: Previously it raises when `name` is already set. This is relaxed
            # because this will be invoked multiple times when submodule is called
            # multiple times.
            if name in self._initializers and self._initializers[name] is not value:
                raise ValueError(
                    f"Initializer '{name}' exists already with a different value."
                )
        elif (
            self != self._parent_torch_script_graph
            and self._parent_torch_script_graph is not None
        ):
            # Only the root graph can have initializers. Add as initializer
            # to root graph, and add as input to current graph.
            self._initializers_inputs_from_parent[
                name
            ] = self._parent_torch_script_graph.add_initializer(name, value)
            self.add_input(value)
            self._initializers_inputs[name] = value

        self._initializers[name] = value
        self.add_input(value)
        self._initializers_inputs[name] = value

    def _get_constant_tensors(self) -> Dict[str, Tensor]:
        input_const = [tensor for tensor in self._inputs if not tensor.is_fake]
        output_const = [tensor for tensor in self._outputs if not tensor.is_fake]
        node_related_const = [
            tensor
            for node in self._nodes
            for tensor in (node.inputs + node.outputs)
            if not tensor.is_fake
        ]
        return {
            tensor.name: tensor for tensor in input_const + output_const + node_related_const
        }

    @property
    def initializers(self) -> Dict[str, Tensor]:
        return self._initializer or self._get_constant_tensors()

    @runtime_typing.checked
    def _fetch_function_proto_dict(
        self, opset_version: int
    ) -> Mapping[Tuple[str, str], onnx.FunctionProto]:
        function_proto_dict: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        # Fetch local function protos. E.g., local functions representing module calls.
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_proto_dict.update(
                sub_torch_script_graph._fetch_function_proto_dict(opset_version)
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

    def _to_onnx_function(self, opset_version: int, name: str) -> onnx.FunctionProto:
        pass

    def _to_onnx_graph(self) -> onnx.GraphProto:
        # convert tensor to tensor value info according to the onnx_type
        input_value_infos = [tensor.to_value_info() for tensor in self._inputs]
        output_value_infos = [tensor.to_value_info() for tensor in self._outputs]
        # convert initializer
        initializer = [
            constant_tensor.to_value_info() for constant_tensor in self.initializers.values()
        ]
        # convert node to node proto
        node_value_infos = [node.to_node_proto() for node in self._nodes]
        # convert graph to graph proto
        graph_proto = onnx.helper.make_graph(
            nodes=node_value_infos,
            name=self._name,
            inputs=input_value_infos,
            outputs=output_value_infos,
            initializer=initializer,
            doc_string=self._doc_string,
        )
        return graph_proto

    def to_onnx_model(self, opset_version: int) -> onnx.ModelProto:
        graph_proto = self._to_onnx_graph()

        function_proto_dict: Mapping[
            Tuple[str, str], onnx.FunctionProto
        ] = self._fetch_function_proto_dict(opset_version)

        unique_custom_domains: Dict[str, int] = {}
        for function_proto in function_proto_dict.values():
            # TODO(BowenBao): All local function domain versions are hardcoded as 1.
            unique_custom_domains[function_proto.domain] = 1

        model_proto = onnx.helper.make_model(
            graph_proto,
            producer_name=self._producer_name,
            doc_string="",  # TODO
            functions=list(function_proto_dict.values()),
            opset_imports=opset_version,
            ir_version=8,  # TODO
        )

        # TODO: Do we still need this after using onnx.helper api?
        # `_export_onnx` only exports opset_imports that is visible to it. It does not
        # export opset_imports for nested functions, since it does not have access to
        # them. We manually add them back and merge with existing opset_imports in the
        # model proto.
        while len(model_proto.opset_import) > 0:
            opsetid = model_proto.opset_import.pop()
            unique_custom_domains[opsetid.domain] = opsetid.version
        model_proto.opset_import.extend(
            [
                onnx.helper.make_opsetid(domain, version)
                for domain, version in unique_custom_domains.items()
            ]
        )

        model_proto = onnx.shape_inference.infer_shapes(
            model_proto, check_type=True, strict_mode=False, data_prop=True
        )
        onnx.checker.check_model(model_proto, full_check=True)
        return model_proto

    # TODO(titaiwang)
    @runtime_typing.checked
    def from_onnx(self, model: onnx.ModelProto):
        pass


# TODO: Deprecate and use `add_node` instead?
class GraphEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, graph: Graph):
        self._graph: Graph = graph

    @property
    def graph(self) -> Graph:
        return self._graph

    def eval(self, schema, inputs, attributes):
        n_outputs = evaluator.compute_num_outputs(schema, inputs, attributes)
        node = Node.from_op_schema(schema, inputs, attributes, n_outputs)
        return self._graph.add_node(node)

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
        # TODO(titaiwang): DO we still need this?
        for name, value in attributes.items():
            param = name_to_schema[name]
            # Cast int to float if needed
            if param.type in {float, "float"}:
                # FIXME(justinchuby): Create invariant on the type of param.type to simplify this
                attributes[name] = float(value)
        # NOTE: Initialize node
        node = Node.from_function(
            function, inputs, attributes, n_outputs=len(function.function_ir.outputs)
        )
        return self._graph.add_node(node)
