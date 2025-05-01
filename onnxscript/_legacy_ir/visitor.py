# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa: TID251
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Sequence

import numpy as np
import onnx

import onnxscript._legacy_ir as ir
from onnxscript.utils.utils import (
    get_initializer_type,
    is_control_flow_op,
    normalize_domain,
)

logger = logging.getLogger(__name__)


def _override_inferred_value_type_with_symbolic_value_type(
    symbolic_value: ir.Value | None,
    inferred_value: ir.Value | None,
) -> ir.Value | None:
    if inferred_value is not None and symbolic_value is not None:
        inferred_value.type = symbolic_value.type
    if inferred_value is None:
        inferred_value = symbolic_value
    return inferred_value


def is_local_function_node(
    node: onnx.NodeProto, functions: dict[ir.FunctionId, onnx.FunctionProto]
) -> bool:
    return ir.get_function_id_from_node(node) in functions


class FunctionShapeEnv:
    def __init__(self):
        # Mapping from (domain, function_name, overload) to {value_name: ir_value}
        self._function_values: dict[ir.FunctionId, dict[str, ir.Value]] = {}

    def load_from_model_proto(self, model_proto: onnx.ModelProto) -> None:
        for value_info in model_proto.graph.value_info:
            self.load_from_value_info(value_info)

    def save_to_model_proto(self, model_proto: onnx.ModelProto) -> None:
        for (
            domain,
            function_name,
            overload,
        ), named_ir_values in self._function_values.items():
            for ir_value in named_ir_values.values():
                if (
                    value_info := self.save_to_value_info(
                        ir_value, domain, function_name, overload
                    )
                ) is not None:
                    model_proto.graph.value_info.append(value_info)

    def load_from_value_info(self, value_info: onnx.ValueInfoProto) -> None:
        function_id, ir_value = self.process_value_info(value_info)
        if function_id is not None:
            logger.debug(
                "Loads torch symbolic value info '%s'.",
                value_info.name,
            )
            self._function_values.setdefault(function_id, {})[ir_value.name] = ir_value

    def process_value_info(
        self, value_info: onnx.ValueInfoProto
    ) -> tuple[ir.FunctionId | None, ir.Value]:
        name = value_info.name
        if len(splits := name.split("/")) == 2:
            # Experimental function value info format.
            # To be deprecated after ONNX 1.16, where value_info is introduced in FunctionProto.
            function_id, value_name = splits
            splits = function_id.split("::")
            domain, function_name = splits[0], splits[1]
            # 'overload' is introduced in ONNX 1.16, consider it as empty string prior to that.
            # The code is for future proof, in case overload is encoded in this format.
            overload = ""
            if len(splits) == 3:
                overload = splits[2]
            function_id = (domain, function_name, overload)
        else:
            # Standard main graph value info format.
            function_id = None
            value_name = name
        return function_id, ir.Value(name=value_name, type=value_info.type)

    def save_to_value_info(
        self, value: ir.Value, domain: str, function_name: str, overload: str
    ) -> onnx.ValueInfoProto | None:
        if overload != "":
            raise NotImplementedError("Overload is not supported yet.")
        function_id = f"{domain}::{function_name}"

        if value.type is not None:
            return onnx.helper.make_value_info(f"{function_id}/{value.name}", value.type)
        return None

    def lookup(self, function: onnx.FunctionProto, value_name: str) -> ir.Value | None:
        """Lookup ir value of 'value_name' inside 'function'."""
        function_id = ir.get_function_id(function)
        function_values = self._function_values.get(function_id)
        if function_values is None or (ir_value := function_values.get(value_name)) is None:
            logger.debug(
                "Lookup Missed %s torch symbolic value info in function %s::%s.",
                value_name,
                function.domain,
                function.name,
            )
            return None
        logger.debug(
            "Lookup found %s torch symbolic value info in function %s::%s.",
            value_name,
            function.domain,
            function.name,
        )
        return ir_value

    def bind(self, value: ir.Value, domain: str, function_name: str, overload: str) -> None:
        """Bind ir value 'value' to 'value_name' inside 'function'."""
        function_id = (domain, function_name, overload)
        self._function_values.setdefault(function_id, {})[value.name] = value

    def get_ir_values(self, function: onnx.FunctionProto) -> dict[str, ir.Value]:
        """Get all ir values inside 'function'."""
        function_id = ir.get_function_id(function)
        return self._function_values.get(function_id, {})


class SubScope:
    values: dict[str, ir.Value]
    ref_attributes: dict[str, onnx.AttributeProto]
    owner: onnx.GraphProto | onnx.FunctionProto

    def __init__(self, owner: onnx.GraphProto | onnx.FunctionProto):
        self.values = {}
        self.ref_attributes = {}
        self.owner = owner

    def lookup(self, name: str) -> ir.Value | None:
        return self.values.get(name)

    def bind(self, name: str, value: ir.Value) -> None:
        self.values[name] = value

    def lookup_ref_attribute(self, ref_attr_name: str) -> onnx.AttributeProto | None:
        return self.ref_attributes.get(ref_attr_name)

    def bind_ref_attribute(self, ref_attr_name: str, attr: onnx.AttributeProto) -> None:
        self.ref_attributes[ref_attr_name] = attr

    def readable_strs(self, indent: int = 0) -> list[str]:
        indent_str = " " * indent
        strs = []
        if isinstance(self.owner, onnx.GraphProto):
            strs.append(f"Graph {self.owner.name}:")
        else:
            strs.append(f"Function {self.owner.name}:")
        strs.append("  ir.Values:")
        for name, value in self.values.items():
            strs.append(f"    {name}: {value}")
        strs.append("  RefAttributes:")
        for name, attr in self.ref_attributes.items():
            strs.append(f"    {name}: {attr}")

        return [f"{indent_str}{s}" for s in strs]

    def __str__(self) -> str:
        return "\n".join(self.readable_strs())


@dataclasses.dataclass
class Scope:
    _sub_scopes: list[SubScope] = dataclasses.field(default_factory=list)

    def lookup(self, name: str) -> ir.Value | None:
        """Lookup value by name from all SubScopes."""
        for sub_scope in reversed(self._sub_scopes):
            if (result := sub_scope.lookup(name)) is not None:
                return result
        return None

    def bind(self, name: str, value: ir.Value) -> None:
        """Bind value to name in the most recent SubScope."""
        if name == "":
            raise ValueError("Cannot bind to empty name.")
        if value is None:
            raise ValueError(f"Cannot bind None to value {name}.")
        self._sub_scopes[-1].bind(name, value)

    def lookup_or_create(self, name: str) -> ir.Value:
        """Lookup value by name from all SubScopes. If not found, create a new one in most recent SubScope."""
        if name == "":
            raise ValueError("Cannot lookup or create empty name.")
        for sub_scope in reversed(self._sub_scopes):
            if (result := sub_scope.lookup(name)) is not None:
                return result
        value = ir.Value(name=name)
        self.bind(name, value)
        return value

    def lookup_ref_attribute(self, ref_attr_name: str) -> onnx.AttributeProto | None:
        for sub_scope in reversed(self._sub_scopes):
            if (result := sub_scope.lookup_ref_attribute(ref_attr_name)) is not None:
                return result
        return None

    def bind_ref_attribute(self, ref_attr_name: str, attr: onnx.AttributeProto) -> None:
        self._sub_scopes[-1].bind_ref_attribute(ref_attr_name, attr)

    def enter_sub_scope(self, owner: onnx.GraphProto) -> None:
        self._sub_scopes.append(SubScope(owner))

    def exit_sub_scope(self) -> SubScope:
        return self._sub_scopes.pop()

    def current_function_scope(self) -> SubScope | None:
        if len(self._sub_scopes) == 0:
            return None
        if isinstance(self._sub_scopes[0].owner, onnx.FunctionProto):
            return self._sub_scopes[0]
        return None

    def current_function(self) -> onnx.FunctionProto | None:
        current_function_scope = self.current_function_scope()
        if current_function_scope is not None:
            return current_function_scope.owner
        return None

    def current_graph(self) -> onnx.GraphProto | None:
        for sub_scope in reversed(self._sub_scopes):
            if isinstance(sub_scope.owner, onnx.GraphProto):
                return sub_scope.owner
        return None

    def readable_strs(self, indent: int = 0) -> list[str]:
        indent_str = " " * indent
        strs = []
        for i, sub_scope in enumerate(self._sub_scopes):
            strs.append(f"SubScope {i}:")
            strs.extend(sub_scope.readable_strs(indent=indent + 2))
        return [f"{indent_str}{s}" for s in strs]

    def __str__(self) -> str:
        return "\n".join(self.readable_strs())


@dataclasses.dataclass
class ScopeStack:
    """Stack of scopes.

    Each Scope represents statically-nested SubScopes (where inner SubScopes can access names defined in outer SubScopes)
    produced by subgraphs (occurring as attribute values), except for the first SubScope which could be produced by a function.
    With a ScopeStack, there is no such possibility of referencing variables defined higher up in the stack by name.
    Instead, it is meant to represent a sequence of (nested) function-calls. Each entry in the stack (except the outermost)
    represents a call to a function.

    Thus, we would use a ScopeStack for a context-sensitive analysis (where we recursively process a called function).
    For a context-insensitive analysis, we would only need a Scope (where we recursively process subgraphs).

    To debug, `print(scope_stack)` will print the scope structure as well as the info stored
    in each scope.
    """

    _scopes: list[Scope] = dataclasses.field(default_factory=lambda: [Scope()])

    def current_scope(self) -> Scope:
        return self._scopes[-1]

    def lookup(self, name: str) -> ir.Value | None:
        """Lookup value by name from the current Scope."""
        return self.current_scope().lookup(name)

    def bind(self, name: str, value: ir.Value) -> None:
        """Bind value to name in the current Scope."""
        self.current_scope().bind(name, value)

    def lookup_or_create(self, name: str) -> ir.Value:
        """Lookup value by name from the current Scope. If not found, create a new one."""
        return self.current_scope().lookup_or_create(name)

    def lookup_ref_attribute(self, ref_attr_name: str) -> onnx.AttributeProto | None:
        return self.current_scope().lookup_ref_attribute(ref_attr_name)

    def bind_ref_attribute(self, ref_attr_name: str, attr: onnx.AttributeProto) -> None:
        self.current_scope().bind_ref_attribute(ref_attr_name, attr)

    def enter_graph_scope(self, graph: onnx.GraphProto) -> None:
        self.current_scope().enter_sub_scope(graph)

    def exit_graph_scope(self) -> SubScope:
        sub_scope = self.current_scope().exit_sub_scope()
        assert isinstance(sub_scope.owner, onnx.GraphProto), "Expected graph scope."
        return sub_scope

    def enter_function_scope(self, function: onnx.FunctionProto) -> None:
        self._scopes.append(Scope())
        self.current_scope().enter_sub_scope(function)

    def exit_function_scope(self) -> SubScope:
        sub_scope = self.current_scope().exit_sub_scope()
        assert isinstance(sub_scope.owner, onnx.FunctionProto), "Expected function scope."
        self._scopes.pop()
        return sub_scope

    def current_function(self) -> onnx.FunctionProto | None:
        return self.current_scope().current_function()

    def current_graph(self) -> onnx.GraphProto | None:
        return self.current_scope().current_graph()

    def __str__(self) -> str:
        strs = ["ScopeStach:"]
        for i, scope in enumerate(self._scopes):
            strs.append(f"  Scope {i}:")
            strs.extend(scope.readable_strs(indent=2))
        return "\n".join(strs)


class ProtoVisitorCore:
    def visit_model(self, model: onnx.ModelProto):
        self.process_model(model)
        for opset in model.opset_import:
            self.process_opset_import(opset)
        self.visit_graph(model.graph)
        for function in model.functions:
            self.visit_function(function)

    def process_model(self, model: onnx.ModelProto):
        pass

    def process_opset_import(self, opset: onnx.OperatorSetIdProto):
        pass

    def visit_graph(self, graph: onnx.GraphProto):
        self.enter_scope(graph)
        self.process_graph(graph)
        for input in graph.input:
            self.process_graph_input(input)
        for init in graph.initializer:
            self.process_initializer(init)
        for value_info in graph.value_info:
            self.process_value_info(value_info)
        for node in graph.node:
            self.visit_node(node)
        for output in graph.output:
            self.process_graph_output(output)
        self.exit_scope(graph)

    def visit_function(self, function: onnx.FunctionProto):
        self.enter_function_scope(function)
        self.process_function(function)
        for input in function.input:
            self.process_function_input(input)
        for node in function.node:
            self.visit_node(node)
        for output in function.output:
            self.process_function_output(output)
        self.exit_function_scope(function)

    def process_function_input(self, input: str):
        pass

    def process_function_output(self, output: str):
        pass

    def process_function(self, function: onnx.FunctionProto):
        pass

    def enter_function_scope(self, function: onnx.FunctionProto):
        pass

    def exit_function_scope(self, function: onnx.FunctionProto) -> SubScope:
        pass

    def enter_scope(self, graph: onnx.GraphProto):
        pass

    def process_graph(self, graph: onnx.GraphProto):
        pass

    def exit_scope(self, graph: onnx.GraphProto) -> SubScope:
        pass

    def process_graph_input(self, input: onnx.ValueInfoProto):
        pass

    def process_initializer(self, init: onnx.TensorProto):
        pass

    def process_value_info(self, value_info: onnx.ValueInfoProto):
        pass

    def visit_node(self, node: onnx.NodeProto):
        self.process_node(node)
        for attr in node.attribute:
            self.visit_attribute(attr)

    def process_node(self, node: onnx.NodeProto) -> Sequence[onnx.NodeProto] | None:
        pass

    def process_graph_output(self, output: onnx.ValueInfoProto):
        pass

    def visit_attribute(self, attr: onnx.AttributeProto):
        self.process_attribute(attr)
        if attr.HasField("g"):
            self.visit_graph(attr.g)
        elif len(attr.graphs) > 0:
            for graph in attr.graphs:
                self.visit_graph(graph)

    def process_attribute(self, attr: onnx.AttributeProto):
        pass


class ProtoVisitor(ProtoVisitorCore):
    def __init__(
        self, external_data_folder: str = "", *, do_shape_inference: bool = False
    ) -> None:
        super().__init__()
        self.scopes = ScopeStack()
        self.function_shape_env = FunctionShapeEnv()
        self.version_map = {}  # Map from domain to version
        self.do_shape_inference = do_shape_inference
        self.external_data_folder = external_data_folder
        self.modified = False

    def process_opset_import(self, opset: onnx.OperatorSetIdProto):
        domain = normalize_domain(opset.domain)
        self.version_map[domain] = opset.version

    def lookup_version(self, domain: str) -> int:
        domain = normalize_domain(domain)
        return self.version_map.get(domain, 1)  # TODO: handle missing domain

    def lookup(self, name: str) -> ir.Value | None:
        if name == "":
            return None
        if (result := self.scopes.lookup(name)) is None:
            logger.debug("Lookup value %s unfound.", name)
            raise ValueError(
                f"Undefined variable {name}.\n"
                f"Available variables: {self.scopes.current_scope()}"
            )
        logger.debug("Lookup value %s. Shape %s", name, result.tensor_shape_proto())
        return result

    def bind(self, name: str, value: ir.Value) -> None:
        logger.debug("Binding value %s. Shape %s", name, value.tensor_shape_proto())
        self.scopes.bind(name, value)

    def lookup_or_create(self, name: str) -> ir.Value:
        return self.scopes.lookup_or_create(name)

    def has_input(self, node: onnx.NodeProto, index: int) -> bool:
        return index < len(node.input) and node.input[index] != ""

    # TODO: Cleanup handling of undefined variables. May fail in some of methods below.

    def get_input(self, node: onnx.NodeProto, index: int) -> ir.Value | None:
        if index < len(node.input):
            return self.lookup(node.input[index])
        return None

    def input_type(self, node: onnx.NodeProto, index: int) -> onnx.TypeProto | None:
        info = self.get_input(node, index)
        return info.type if info is not None else None

    def input_element_type(self, node: onnx.NodeProto, index: int) -> int | None:
        info = self.get_input(node, index)
        return info.element_type if info is not None else None

    def input_shape(self, node: onnx.NodeProto, index: int) -> onnx.TensorShapeProto | None:
        info = self.get_input(node, index)
        return info.tensor_shape_proto() if info is not None else None

    def input_const_value(self, node: onnx.NodeProto, index: int) -> Any:
        if not self.has_input(node, index):
            return None  # This is treated as a known constant value "None"
        info = self.get_input(node, index)
        return info.value

    def has_output(self, node: onnx.NodeProto, index: int) -> bool:
        return index < len(node.output) and node.output[index] != ""

    def get_output(self, node: onnx.NodeProto, index: int) -> ir.Value | None:
        if index < len(node.output):
            return self.lookup(node.output[index])
        return None

    def get_input_value(
        self, node: onnx.NodeProto, index: int, default: Any | None = None
    ) -> Any | None:
        info = self.get_input(node, index)
        if info is not None:
            return info.value
        return default

    def get_input_type(
        self, node: onnx.NodeProto, index: int, default: onnx.TypeProto | None = None
    ) -> onnx.TypeProto | None:
        info = self.get_input(node, index)
        if info is not None:
            return info.type
        return default

    def enter_scope(self, graph: onnx.GraphProto):
        logger.debug("enter_scope: graph %s", graph.name)
        self.scopes.enter_graph_scope(graph)

    def exit_scope(self, graph: onnx.GraphProto) -> SubScope:
        logger.debug("exit_scope: graph %s", graph.name)
        return self.scopes.exit_graph_scope()

    def enter_function_scope(self, function: onnx.FunctionProto):
        logger.debug("enter_function_scope: function %s", function.name)
        self.scopes.enter_function_scope(function)
        ir_values = self.function_shape_env.get_ir_values(function)
        for name, ir_value in ir_values.items():
            inferred_ir_value = self.lookup_or_create(name)
            updated_ir_value = _override_inferred_value_type_with_symbolic_value_type(
                ir_value, inferred_ir_value
            )
            self.bind(name, updated_ir_value)

    def exit_function_scope(self, function: onnx.FunctionProto) -> SubScope:
        logger.debug("exit_function_scope: function %s", function.name)
        # Sync ir value back to function_shape_env
        function_scope = self.scopes.exit_function_scope()
        for ir_value in function_scope.values.values():
            self.function_shape_env.bind(ir_value, *ir.get_function_id(function))
        return function_scope

    def process_initializer(self, init: onnx.TensorProto):
        array = onnx.numpy_helper.to_array(init, self.external_data_folder)
        self.bind(
            init.name,
            ir.Value(name=init.name, value=array, type=get_initializer_type(init)),
        )

    def process_graph_input(self, input: onnx.ValueInfoProto):
        self.bind(input.name, ir.Value(name=input.name, type=input.type))

    def process_value_info(self, value_info: onnx.ValueInfoProto):
        logger.debug("process_value_info: %s", value_info)
        value = self.lookup_or_create(value_info.name)
        value.type = value_info.type
        # Populate function shape environment
        self.function_shape_env.load_from_value_info(value_info)

    def process_node(self, node: onnx.NodeProto) -> Sequence[onnx.NodeProto] | None:
        output_types = {}
        if self.do_shape_inference and not is_control_flow_op(node):
            # Control-flow ops are more complicated. Not supported here yet.
            # TODO: handle optional inputs
            def get_constant_value(i: int) -> onnx.TensorProto | None:
                value = self.input_const_value(node, i)
                if isinstance(value, np.ndarray) and value.size < 20:
                    return onnx.numpy_helper.from_array(value, node.input[i])
                return None

            input_types = {x: self.input_type(node, i) for i, x in enumerate(node.input)}
            input_data = {x: get_constant_value(i) for i, x in enumerate(node.input)}
            input_data = {k: v for k, v in input_data.items() if v is not None}
            if any(t is None for t in input_types.values()):
                logger.debug(
                    "Skipping shape inference for node %s due to missing input type.",
                    node.name,
                )
            else:
                # TODO: pass in constant values, ir_version
                try:
                    schema = onnx.defs.get_schema(
                        node.op_type, self.lookup_version(node.domain), node.domain
                    )
                    output_types = onnx.shape_inference.infer_node_outputs(
                        schema, node, input_types, input_data
                    )
                except Exception as e:
                    logger.debug(
                        "Skipping shape inference for node %s due to exception: %s",
                        node.name,
                        e,
                    )

        for output in node.output:
            if output == "":
                continue
            info = self.lookup_or_create(output)
            if output in output_types:
                if info.type is not None:
                    if (
                        info.type.tensor_type.elem_type
                        != output_types[output].tensor_type.elem_type
                    ):
                        logger.warning(
                            "Overriding existing type %s with inferred type %s for %s",
                            info.type,
                            output_types[output],
                            output,
                        )
                # TODO: merge types
                info.type = output_types[output]


class ProtoTransformer(ProtoVisitor):
    # TODO(lowpri) Practically this is useless.
    # Subgraph only exist in 'if' nodes. 'if' nodes only exist in torchlib functions.
    # There is no pre-existing value_info in torchlib functions.
    # def exit_scope(self, graph: onnx.GraphProto) -> SubScope:
    #     # Also sync updated ir values back to value_info in graph.
    #     sub_scope = super().exit_scope(graph)

    def visit_node(self, node: onnx.NodeProto) -> list[onnx.NodeProto] | None:
        replacement = self.process_node(node)
        logger.debug(
            "visit_node: %s::%s %s replacement %s",
            node.domain,
            node.op_type,
            node.name,
            "found" if replacement is not None else "missed",
        )
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attribute:
                self.visit_attribute(attr)
            return None
        else:
            self.modified = True
            # We recursively visit the replacement nodes.
            result = []
            for newnode in replacement:
                n = self.visit_node(newnode)
                if n is not None:
                    result.extend(n)
                else:
                    result.append(newnode)
            return result

    def visit_graph(self, graph: onnx.GraphProto) -> dict[str, ir.Value]:
        self.enter_scope(graph)
        self.process_graph(graph)
        for input in graph.input:
            self.process_graph_input(input)
        for init in graph.initializer:
            self.process_initializer(init)
        for value_info in graph.value_info:
            self.process_value_info(value_info)
        updates = []
        nodes = graph.node
        for i, node in enumerate(nodes):
            replacement = self.visit_node(node)
            if replacement is not None:
                updates.append((i, replacement))
        for i, replacement in reversed(updates):
            old_node_name = nodes[i].name
            del nodes[i]
            for newnode in reversed(replacement):
                logger.debug(
                    "Replacement node %s for %s. Size %s",
                    newnode.name,
                    old_node_name,
                    newnode.ByteSize(),
                )
                nodes.insert(i, newnode)
        for output in graph.output:
            self.process_graph_output(output)
        return self.exit_scope(graph)


class FunctionCallsiteAnalysis(ProtoVisitor):
    """Collects the callsites of each function."""

    def __init__(self):
        super().__init__()
        self.functions: dict[ir.FunctionId, onnx.FunctionProto] = {}
        self.function_calls: dict[ir.FunctionId, list[onnx.NodeProto]] = {}

    def visit_function(self, function: onnx.FunctionProto):
        # Do not visit function via model.functions.
        # Only visit function at callsites.
        # The purpose of this analysis is to collect the callsites of each function.
        pass

    def visit_node(self, node: onnx.NodeProto) -> None:
        if is_local_function_node(node, self.functions):
            function_id = ir.get_function_id_from_node(node)
            self.function_calls.setdefault(function_id, []).append(node)
            for subnode in self.functions[function_id].node:
                self.visit_node(subnode)

    def visit_model(self, model: onnx.ModelProto) -> None:
        for function in model.functions:
            self.functions[ir.get_function_id(function)] = function

        super().visit_model(model)


class FunctionRenamer:
    _POSTFIX_FORMAT = "{name}|{postfix}_{count}"

    def __init__(self, postfix="folded"):
        self._function_key_to_instance_count = {}
        self._postfix = postfix

    def rename(self, function: onnx.FunctionProto) -> None:
        domain = function.domain
        name = function.name
        key = (domain, name)
        self._function_key_to_instance_count.setdefault(key, 0)
        function.name = self._POSTFIX_FORMAT.format(
            name=name,
            postfix=self._postfix,
            count=self._function_key_to_instance_count[key],
        )
        self._function_key_to_instance_count[key] += 1


class FunctionCallsiteProtoTransformer(ProtoTransformer):
    """Unlike other base visitors, this is a special visitor that visits functions at their callsite.

    This allows transforming and constructing specialized functions based on callsite context.
    """

    _functions: dict[ir.FunctionId, onnx.FunctionProto]
    _function_callsites: dict[ir.FunctionId, list[onnx.NodeProto]]
    _new_functions: list[onnx.FunctionProto]
    _function_renamer: FunctionRenamer

    def _gather_function_metadata(self, model: onnx.ModelProto):
        analysis = FunctionCallsiteAnalysis()
        analysis.visit_model(model)
        self._functions = analysis.functions
        self._function_callsites = analysis.function_calls
        self._new_functions = []
        self._function_renamer = FunctionRenamer()

    def process_function_outputs(self, function: onnx.FunctionProto) -> bool:
        """Process function outputs.

        This method is called when a function is visited at its callsite.

        Returns:
            True if the function outputs are modified.
        """
        del function  # Unused
        return False

    def process_function_node_outputs(
        self,
        node: onnx.NodeProto,
        function_scope: SubScope,
    ) -> None:
        """Fetch value infos of function output to re-bind them for function node output."""
        function = function_scope.owner
        output_values = [function_scope.lookup(output) for output in function.output]
        for actual_name, formal_value in zip(node.output, output_values):
            if formal_value is None:
                raise RuntimeError(
                    "Missing output %s in function-call to %s",
                    actual_name,
                    node.op_type,
                )
            actual_value = self.lookup_or_create(actual_name)
            actual_value.identity_merge_from(formal_value)
            if logger.level <= logging.INFO:
                logger.info(
                    "Binding outputs for function %s. %s => %s",
                    function.name,
                    actual_value,
                    node.output,
                )

    def lookup_ref_attribute(self, ref_attr_name: str) -> onnx.AttributeProto | None:
        return self.scopes.lookup_ref_attribute(ref_attr_name)

    def bind_ref_attribute(self, ref_attr_name: str, attr: onnx.AttributeProto) -> None:
        self.scopes.bind_ref_attribute(ref_attr_name, attr)

    def visit_model(self, model: onnx.ModelProto):
        self._gather_function_metadata(model)

        self.process_model(model)
        for opset in model.opset_import:
            self.process_opset_import(opset)
        self.visit_graph(model.graph)

        for new_function in self._new_functions:
            model.functions.append(new_function)

        self.function_shape_env.save_to_model_proto(model)

    def visit_node(self, node: onnx.NodeProto) -> list[onnx.NodeProto] | None:
        if is_local_function_node(node, self._functions):
            function_id = ir.get_function_id_from_node(node)
            if function_id not in self._functions:
                # Do not recursively visit new functions.
                return None
            replacement, _ = self.process_function_node(node)
        else:
            replacement = self.process_node(node)
        logger.debug(
            "visit_node: %s::%s %s replacement %s",
            node.domain,
            node.op_type,
            node.name,
            "found" if replacement is not None else "missed",
        )
        if replacement is None:
            # No change. Process attributes.
            for attr in node.attribute:
                self.visit_attribute(attr)
            return None
        else:
            self.modified = True
            # We recursively visit the replacement nodes.
            result = []
            for newnode in replacement:
                n = self.visit_node(newnode)
                if n is not None:
                    result.extend(n)
                else:
                    result.append(newnode)
            return result

    def process_function_node(
        self, node: onnx.NodeProto
    ) -> tuple[list[onnx.NodeProto] | None, onnx.FunctionProto | None]:
        function_id = ir.get_function_id_from_node(node)
        function = self._functions[function_id]

        is_unique_callsite = len(self._function_callsites[function_id]) == 1
        if not is_unique_callsite:
            mutable_function = onnx.FunctionProto()
            mutable_function.CopyFrom(function)
        else:
            mutable_function = function

        logger.info("Visit function %s node %s", function_id, node.name)
        actual_input_value_infos = [self.lookup(input) for input in node.input]
        # Handle omitted inputs, these are considered optional inputs of the function.
        actual_input_value_infos.extend(
            [None] * (len(function.input) - len(actual_input_value_infos))
        )
        ref_attributes = {
            attr_proto.name: self.lookup_ref_attribute(attr_proto.ref_attr_name)
            for attr_proto in node.attribute
            if attr_proto.ref_attr_name
        }

        self.enter_function_scope(mutable_function)
        if logger.level <= logging.INFO:
            printable_actual_input_value_infos = [str(x) for x in actual_input_value_infos]
            logger.info(
                "Actual input value infos: %s",
                printable_actual_input_value_infos,
            )
        logger.info("Enter function scope: %s", self.scopes.current_scope())

        logger.debug("Binding inputs for function %s", function.name)
        for actual_input_value_info, formal_input in zip(
            actual_input_value_infos, function.input
        ):
            formal_info = ir.Value(formal_input)
            if actual_input_value_info is not None:
                formal_info.identity_merge_from(actual_input_value_info)
            self.bind(formal_input, formal_info)

        for attr_proto in function.attribute_proto:
            # Default value of function attributes.
            self.bind_ref_attribute(attr_proto.name, attr_proto)

        for attr_proto in node.attribute:
            if attr_proto.ref_attr_name:
                concrete_attribute = ref_attributes.get(attr_proto.name)
                if concrete_attribute is None:
                    continue
                self.bind_ref_attribute(attr_proto.name, concrete_attribute)
            else:
                self.bind_ref_attribute(attr_proto.name, attr_proto)

        # Visit inner function nodes.
        node_updates: list[tuple[int, list[onnx.NodeProto]]] = []
        nodes = mutable_function.node
        for i, inner_node in enumerate(nodes):
            replacement = self.visit_node(inner_node)
            if replacement is not None:
                node_updates.append((i, replacement))
        for i, replacement in reversed(node_updates):
            old_node_name = nodes[i].name
            old_node_op_type = nodes[i].op_type
            del nodes[i]
            for newnode in reversed(replacement):
                logger.debug(
                    "Replacement node inside function %s: %s for %s %s. Size %s",
                    node.name,
                    newnode.output,
                    old_node_name,
                    old_node_op_type,
                    newnode.ByteSize(),
                )
                nodes.insert(i, newnode)
        added_domains = set()
        del mutable_function.opset_import[:]
        for inner_node in nodes:
            # Update opset_import if needed.
            if inner_node.domain not in added_domains:
                version = self.lookup_version(inner_node.domain)
                mutable_function.opset_import.append(
                    onnx.OperatorSetIdProto(domain=inner_node.domain, version=version)
                )
                added_domains.add(inner_node.domain)

        output_updates = self.process_function_outputs(mutable_function)

        is_new_function = not is_unique_callsite and (node_updates or output_updates)
        if is_new_function:
            self._new_functions.append(mutable_function)
            self._function_renamer.rename(mutable_function)
            node.op_type = mutable_function.name

        function_scope = self.exit_function_scope(mutable_function)

        self.process_function_node_outputs(node, function_scope)

        logger.info("Exit function scope: %s", function_scope)
        logger.info("Exit function %s node %s", function_id, node.name)

        if is_new_function:
            return [node], mutable_function
        return None, None
