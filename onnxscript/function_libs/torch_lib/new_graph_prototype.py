# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

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
from onnxscript import evaluator, onnx_opset
from onnxscript import tensor as onnxscript_tensor
from onnxscript._internal import runtime_typing

__all__ = [
    "Graph",
    "GraphEvaluator",
    "GraphTensor",
]


ValidArgumentType: TypeAlias = Union[
    "GraphTensor",
    Sequence["GraphTensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]


class GraphTensor(onnxscript_tensor.Tensor):
    """An implementation of ONNX Tensors, based on a wrapper around numpy arrays.
    Serves to define overloaded ops with an ONNX/ONNXScript semantics.
    """

    def __init__(
        self,
        torch_tensor: torch.Tensor,
        name: str,
        opset=None,
        onnx_type: str = "tensor_type",
    ):
        super().__init__(None)
        self._name = name
        self._torch_tensor = torch_tensor
        self._shape: tuple[int, ...] = torch_tensor.shape
        self._dtype: torch.dtype = torch_tensor.dtype
        # FIXME(justinhuby): Create a better way to determine the opset version
        self._opset: Any = opset or onnx_opset.opset18

        # TODO: Type class?
        # NOTE: This is used to determine the type of the tensor
        # when we convert it to onnx graph
        if onnx_type not in {"tensor_type", "sequence_type", "optional_type"}:
            raise ValueError(
                f"Invalid onnx type: {onnx_type}, must be one of the three: tensor_type, sequence_type, optional_type"
            )
        self._onnx_type = onnx_type

    def to_tensor_proto(self):
        pass

    def to_value_info(self) -> onnx.ValueInfoProto:
        # TODO: support more types?
        if self.onnx_type == "tensot_type":
            return onnx.helper.make_tensor_value_info(self.name, self.onnx_dtype, self.shape)
        if self.onnx_type == "sequence_type":
            return onnx.helper.make_tensor_sequence_value_info(
                self.name, self.onnx_dtype, self.shape
            )
        if self.onnx_type == "optional_type":
            # NOTE: no `make_optional_value_info` API
            element_type = onnx.helper.make_tensor_type_proto(self.onnx_dtype, self.shape)
            optional_type_proto = onnx.helper.make_optional_type_proto(element_type)
            return onnx.helper.make_value_info(self.name, optional_type_proto)
        return onnx.helper.make_empty_tensor_value_info(self.name)

    # TODO: better way to find out if it's a fake tensor?
    @property
    def is_fake(self) -> bool:
        from torch._subclasses import (  # pylint: disable=import-outside-toplevel
            fake_tensor,
        )

        return isinstance(self._torch_tensor, fake_tensor.FakeTensor)

    @property
    def np_dtype(self) -> np.dtype:
        from torch.onnx._internal.fx import (  # pylint: disable=import-outside-toplevel
            type_utils,
        )

        return type_utils.from_torch_dtype_to_numpy_dtype(self.dtype)

    @property
    def onnx_type(self) -> str:
        return self._onnx_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> torch.Tensor:  # type: ignore[override]
        return self._torch_tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return self._dtype

    @property
    def onnx_dtype(self) -> int:
        return (
            onnx.helper.np_dtype_to_tensor_dtype(self.np_dtype)
            if self.dtype is not None
            else None
        )


###################################################################################################


class Node:
    def __init__(
        self,
        namespace: str,
        op_name: str,
        inputs: List[GraphTensor],
        attributes: Dict[str, ValidArgumentType],
        outputs: List[GraphTensor],
    ):
        self._inputs = inputs
        self._outputs = outputs
        self._attributes = attributes
        self._namespace = namespace
        self._op_name = op_name

    @property
    def inputs(self) -> List[GraphTensor]:
        return self._inputs

    @property
    def outputs(self) -> List[GraphTensor]:
        return self._outputs

    @property
    def attributes(self) -> Dict[str, ValidArgumentType]:
        return self._attributes

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def op_name(self) -> str:
        return self._op_name

    def to_node_proto(self) -> onnx.NodeProto:
        onnx_node = onnx.helper.make_node(
            self.op_name,
            inputs=[t.name for t in self.inputs],
            outputs=[t.name for t in self.outputs],  # TODO: output names
            domain=self.namespace,
        )
        # TODO: process value?
        for k, v in self.attributes.items():
            onnx_node.attribute.append(onnx.helper.make_attribute(k, v))
        return onnx_node

    @classmethod
    def from_op_schema(
        cls,
        op_schema: onnx.defs.OpSchema,
        inputs: List[GraphTensor],
        attributes: Dict[str, ValidArgumentType],
        outputs: List[GraphTensor],
    ) -> OpNode:
        return OpNode(
            opschema=op_schema,
            inputs=inputs,
            attributes=attributes,
            outputs=outputs,
        )

    @classmethod
    def from_function(
        cls,
        onnx_function: onnxscript.OnnxFunction,
        inputs: List[GraphTensor],
        attributes: Dict[str, ValidArgumentType],
        outputs: List[GraphTensor],
    ) -> FunctionNode:
        return FunctionNode(
            function=onnx_function,
            inputs=inputs,
            attributes=attributes,
            outputs=outputs,
        )

    @classmethod
    def from_module(cls, subgraph: Graph, name: str, inputs: List[GraphTensor]) -> ModuleNode:
        return ModuleNode(
            subgraph=subgraph,
            inputs=inputs,
            outputs=subgraph.outputs,
            name=name,
        )


class OpNode(Node):
    def __init__(
        self,
        opschema: onnx.defs.OpSchema,
        inputs: List[GraphTensor],
        attributes: Dict[str, ValidArgumentType],
        outputs: List[GraphTensor],
    ):
        super().__init__(
            namespace=opschema.domain,
            op_name=opschema.name,
            inputs=inputs,
            attributes=attributes,
            outputs=outputs,
        )
        self._opschema = opschema

    def opschema(self) -> onnx.defs.OpSchema:
        return self._opschema


class FunctionNode(Node):
    def __init__(
        self,
        function: onnxscript.OnnxFunction,
        inputs: List[GraphTensor],
        attributes: Dict[str, ValidArgumentType],
        outputs: List[GraphTensor],
    ):
        super().__init__(
            namespace=function.function_ir.domain,
            op_name=function.name,
            inputs=inputs,
            attributes=attributes,
            outputs=outputs,
        )
        self._function = function

    @property
    def function(self) -> onnxscript.OnnxFunction:
        return self._function


class ModuleNode(Node):
    def __init__(
        self, subgraph: Graph, inputs: List[GraphTensor], outputs: List[GraphTensor], name: str
    ):
        super().__init__(
            namespace=subgraph.domain_name,  # type: ignore[arg-type]
            op_name=name,
            inputs=inputs,
            attributes={},
            outputs=outputs,
        )
        self._subgraph = subgraph

    @property
    def module(self) -> Graph:
        return self._subgraph


class Graph:
    def __init__(
        self,
        producer_name="pytorch",
        parent_graph: Optional[Graph] = None,
        domain_name: Optional[str] = None,
        opset_version: int = 18,
    ) -> None:
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
        self._nodes: List[Node] = []
        self._inputs: List[GraphTensor] = []
        self._outputs: List[GraphTensor] = []
        self._initializers: Dict[str, GraphTensor] = {}
        self._producer_name: str = producer_name
        self._opset_version: int = opset_version
        # TODO: what are these two string for?
        self._doc_string: str = ""
        self._name: str = ""
        # NOTE: below are used by splitting subgraphs
        # Mapping from intializer name to input(ensor).
        self._initializers_inputs: Dict[str, GraphTensor] = {}
        # Mapping from intializer name to input(TorchScriptTensor) from parent graph.
        self._initializers_inputs_from_parent: Dict[str, GraphTensor] = {}
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
    def opset_version(self) -> int:
        return self._opset_version

    @property
    def producer_name(self) -> str:
        return self._producer_name

    @property
    def outputs(self) -> List[GraphTensor]:
        return self._outputs

    @property
    def domain_name(self) -> Optional[str]:
        return self._domain_name

    @property
    def inputs(self) -> List[GraphTensor]:
        return self._inputs

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def doc_string(self) -> str:
        return self._doc_string

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
    def add_input(self, tensor: GraphTensor):
        self._inputs.append(tensor)

    @runtime_typing.checked
    def add_output(self, tensor: GraphTensor):
        self._outputs.append(tensor)

    # TODO(titaiwang): this function is not friendly to users when
    # they want to build the graph from scratch, as they need to
    # manually set the value to initializers, so we need another
    # method to get all initializers from the graph `self.initializers`
    @runtime_typing.checked
    def add_initializer(self, name: str, value: GraphTensor):
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

    def _get_constant_tensors(self) -> Dict[str, GraphTensor]:
        input_const = [tensor for tensor in self._inputs if not tensor.is_fake]
        output_const = [tensor for tensor in self._outputs if not tensor.is_fake]
        node_related_const = [
            tensor
            for node in self._nodes
            for tensor in (node.inputs + node.outputs)  # TODO: outputs?
            if not tensor.is_fake
        ]
        return {
            tensor.name: tensor for tensor in input_const + output_const + node_related_const
        }

    @property
    def initializers(self) -> Dict[str, GraphTensor]:
        return self._initializers or self._get_constant_tensors()

    @runtime_typing.checked
    def fetch_function_proto_dict(self) -> Mapping[Tuple[str, str], onnx.FunctionProto]:
        function_proto_dict: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        # Fetch local function protos. E.g., local functions representing module calls.
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_proto_dict.update(sub_torch_script_graph.fetch_function_proto_dict())
            domain = sub_torch_script_graph.domain_name
            assert domain is not None
            name_domain = (
                sub_graph_name,
                domain,
            )
            assert (
                name_domain not in function_proto_dict
            ), f"Sub graph name already exists. {name_domain}"
            # module nodes are not added to the graph, so we need to add them here
            function_proto_dict[name_domain] = sub_torch_script_graph.to_onnx_function(
                sub_graph_name
            )
        # Fetch torchlib function protos.
        for name_domain, function in self._function_store.items():
            function_proto_dict[name_domain] = function.to_function_proto()
        return function_proto_dict

    def to_onnx_function(self, name: str) -> onnx.FunctionProto:
        domain = self.domain_name
        node_value_infos = [node.to_node_proto() for node in self._nodes]
        if domain is None:
            raise RuntimeError("Domain name is not set.")
        onnx_function = onnx.helper.make_function(
            domain=domain,
            fname=name,
            inputs=[input.name for input in self.inputs],
            outputs=[output.name for output in self.outputs],
            nodes=node_value_infos,
            opset_imports=[
                onnx.helper.make_opsetid(domain, self.opset_version)
            ],  # TODO: correct?
            doc_string=self.doc_string,
        )
        # TODO: onnx.checker.check_function(onnx_function)?
        return onnx_function

    def _to_onnx_graph(self) -> onnx.GraphProto:
        # convert tensor to tensor value info according to the onnx_type
        input_value_infos = [tensor.to_value_info() for tensor in self._inputs]
        output_value_infos = [tensor.to_value_info() for tensor in self._outputs]
        # convert initializer
        initializer = [
            constant_tensor.to_tensor_proto() for constant_tensor in self.initializers.values()
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
            doc_string=self.doc_string,
        )
        return graph_proto

    def to_onnx_model(self) -> onnx.ModelProto:
        graph_proto = self._to_onnx_graph()

        function_proto_dict: Mapping[
            Tuple[str, str], onnx.FunctionProto
        ] = self.fetch_function_proto_dict()

        unique_custom_domains: Dict[str, int] = {}
        for function_proto in function_proto_dict.values():
            # TODO(BowenBao): All local function domain versions are hardcoded as 1.
            unique_custom_domains[function_proto.domain] = 1

        model_proto = onnx.helper.make_model(
            graph_proto,
            producer_name=self._producer_name,
            doc_string=self.doc_string,
            functions=list(function_proto_dict.values()),
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
# Evaluator is used to evaluate the graph and add nodes to the graph
# Do we want to keep this UX? One of the blocks is that we need to
# define outputs AFTER we define the node, which is not intuitive.
# If we do, we need to support n_outputs in the Node class.
class GraphEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, graph: Graph):
        self._graph: Graph = graph

    @property
    def graph(self) -> Graph:
        return self._graph

    # def eval(self, schema, inputs, attributes):
    #     n_outputs = evaluator.compute_num_outputs(schema, inputs, attributes)
    #     node = Node.from_op_schema(schema, inputs, attributes, outputs)
    #     return self._graph.add_node(node)

    # @runtime_typing.checked
    # def eval_function(  # type: ignore[override]
    #     self,
    #     function: onnxscript.OnnxFunction,
    #     args: Sequence[ValidArgumentType],
    #     kwargs: Mapping[str, ValidArgumentType],
    # ):
    #     # args/kwargs are TorchScriptTensor/python built-in based
    #     param_schemas = function.param_schemas()
    #     (
    #         inputs,
    #         attributes,
    #     ) = param_manipulation.separate_input_attributes_from_arguments(
    #         param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
    #     )
    #     name_to_schema = {param.name: param for param in param_schemas}
    #     # TODO(titaiwang): DO we still need this?
    #     for name, value in attributes.items():
    #         param = name_to_schema[name]
    #         # Cast int to float if needed
    #         if param.type in {float, "float"}:
    #             # FIXME(justinchuby): Create invariant on the type of param.type to simplify this
    #             attributes[name] = float(value)
    #     # NOTE: Initialize node
    #     node = Node.from_function(
    #         function, inputs, attributes, outputs=function.function_ir.outputs
    #     )
    #     return self._graph.add_node(node)
