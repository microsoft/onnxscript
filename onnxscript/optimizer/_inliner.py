# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Implementation of an inliner for onnxscript.ir"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import onnxscript.ir as ir
import onnxscript.ir.convenience as ir_convenience

# A replacement for a node specifies a list of nodes that replaces the original node,
# and a list of values that replaces the original node's outputs.

NodeReplacement = Tuple[Sequence[ir.Node], Sequence[ir.Value]]

# A call stack is a list of identifiers of call sites, where the first element is the
# outermost call site, and the last element is the innermost call site. This is used
# primarily for generating unique names for values in the inlined functions.
CallSiteId = str
CallStack = List[CallSiteId]


def _make_unique_name(name: str, callstack: CallStack, used_names: set[str]) -> str:
    """Generate a unique name from a name, calling-context, and set of used names.

    If there is a name clash, we add a numeric suffix to the name to make
    it unique. We use the same strategy to make node names unique.

    TODO: We can use the callstack in generating a name for a value X in a function
    that is inlined into a graph. This is not yet implemented. Using the full callstack
    leads to very long and hard to read names. Some investigation is needed to find
    a good naming strategy that will produce useful names for debugging.
    """
    candidate = name
    i = 1
    while candidate in used_names:
        i += 1
        candidate = f"{name}_{i}"
    used_names.add(candidate)
    return candidate


class _CopyReplace:
    """Utilities for creating a copy of IR objects with substitutions for attributes/input values."""

    def __init__(
        self,
        inliner: _Inliner,
        attr_map: dict[str, ir.Attr | ir.RefAttr],
        value_map: dict[ir.Value, ir.Value | None],
        metadata_props: dict[str, str],
        call_stack: CallStack,
    ) -> None:
        self._inliner = inliner
        self._value_map = value_map
        self._attr_map = attr_map
        self._metadata_props = metadata_props
        self._call_stack = call_stack

    def clone_value(self, value: ir.Value) -> ir.Value | None:
        if value in self._value_map:
            return self._value_map[value]
        # If the value is not in the value map, it must be a graph input.
        assert value.producer() is None, f"Value {value} has no entry in the value map"
        new_value = ir.Value(
            name=value.name,
            type=value.type,
            shape=value.shape,
            doc_string=value.doc_string,
            const_value=value.const_value,
        )
        self._value_map[value] = new_value
        return new_value

    def clone_optional_value(self, value: ir.Value | None) -> ir.Value | None:
        if value is None:
            return None
        return self.clone_value(value)

    def clone_attr(self, key: str, attr: ir.Attr | ir.RefAttr) -> ir.Attr | ir.RefAttr | None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                graph = self.clone_graph(attr.as_graph())
                return ir.Attr(key, ir.AttributeType.GRAPH, graph, doc_string=attr.doc_string)
            elif attr.type == ir.AttributeType.GRAPHS:
                graphs = [self.clone_graph(graph) for graph in attr.as_graphs()]
                return ir.Attr(
                    key, ir.AttributeType.GRAPHS, graphs, doc_string=attr.doc_string
                )
            return attr
        assert isinstance(attr, ir.RefAttr)
        ref_attr_name = attr.ref_attr_name
        if ref_attr_name in self._attr_map:
            ref_attr = self._attr_map[ref_attr_name]
            if isinstance(ref_attr, ir.Attr):
                return ir.Attr(
                    key, ref_attr.type, ref_attr.value, doc_string=ref_attr.doc_string
                )
            assert isinstance(ref_attr, ir.RefAttr)
            return ir.RefAttr(
                key, ref_attr.ref_attr_name, ref_attr.type, doc_string=ref_attr.doc_string
            )
        # Note that if a function has an attribute-parameter X, and a call (node) to the function
        # has no attribute X, all references to X in nodes inside the function body will be
        # removed. This is just the ONNX representation of optional-attributes.
        return None

    def clone_node(self, node: ir.Node) -> ir.Node:
        new_inputs = [self.clone_optional_value(input) for input in node.inputs]
        new_attributes = [
            new_value
            for key, value in node.attributes.items()
            if (new_value := self.clone_attr(key, value)) is not None
        ]
        new_name = node.name
        if new_name is not None:
            new_name = _make_unique_name(
                new_name, self._call_stack, self._inliner.used_node_names
            )

        new_metadata = {**self._metadata_props, **node.metadata_props}
        # TODO: For now, node metadata overrides callnode metadata if there is a conflict.
        # Do we need to preserve both?

        new_node = ir.Node(
            node.domain,
            node.op_type,
            new_inputs,
            new_attributes,
            overload=node.overload,
            num_outputs=len(node.outputs),
            graph=None,
            name=new_name,
            doc_string=node.doc_string,  # type: ignore
            metadata_props=new_metadata,
        )
        new_outputs = new_node.outputs
        for i, output in enumerate(node.outputs):
            self._value_map[output] = new_outputs[i]
            old_name = output.name if output.name is not None else f"output_{i}"
            new_outputs[i].name = _make_unique_name(
                old_name, self._call_stack, self._inliner.used_value_names
            )

        self._inliner.node_context[new_node] = self._call_stack

        return new_node

    def clone_graph(self, graph: ir.Graph) -> ir.Graph:
        input_values = [self.clone_value(v) for v in graph.inputs]
        nodes = [self.clone_node(node) for node in graph]
        initializers = [self.clone_value(init) for init in graph.initializers.values()]
        output_values = [
            self.clone_value(v) for v in graph.outputs
        ]  # Looks up already cloned values

        return ir.Graph(
            input_values,  # type: ignore
            output_values,  # type: ignore
            nodes=nodes,
            initializers=initializers,  # type: ignore
            doc_string=graph.doc_string,
            opset_imports=graph.opset_imports,
            name=graph.name,
            metadata_props=graph.metadata_props,
        )


def _abbreviate(
    function_ids: Iterable[ir.OperatorIdentifier],
) -> dict[ir.OperatorIdentifier, str]:
    """Create a short unambiguous abbreviation for all function ids."""

    def id_abbreviation(id: ir.OperatorIdentifier) -> str:
        """Create a short unambiguous abbreviation for a function id."""
        domain, name, overload = id
        # Omit the domain, if it remains unambiguous after omitting it.
        if any(x[0] != domain and x[1] == name and x[2] == overload for x in function_ids):
            short_domain = domain + "_"
        else:
            short_domain = ""
        if overload != "":
            return short_domain + name + "_" + overload
        return short_domain + name

    return {id: id_abbreviation(id) for id in function_ids}


class _Inliner:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self._function_id_abbreviations = _abbreviate(self._functions.keys())
        self._opset_imports = model.opset_imports
        self.used_value_names: set[str] = set()
        self.used_node_names: set[str] = set()
        self.node_context: dict[ir.Node, CallStack] = {}

    def _instantiate_call(self, node: ir.Node, call_site_id: CallSiteId) -> NodeReplacement:
        id = node.op_identifier()
        function = self._functions[id]

        # check opset compatibility and update the opset imports
        for key, value in function.opset_imports.items():
            if key not in self._opset_imports:
                self._opset_imports[key] = value
            elif self._opset_imports[key] != value:
                raise ValueError(
                    f"Opset mismatch: {key} {self._opset_imports[key]} != {value}"
                )

        # Identify substitutions for both inputs and attributes of the function:
        attributes: dict[str, ir.Attr | ir.RefAttr] = node.attributes
        default_attr_values = {
            attr.name: attr
            for attr in function.attributes.values()
            if attr.name not in attributes and attr.value is not None
        }
        if default_attr_values:
            attributes = {**attributes, **default_attr_values}
        if any(
            attr.type == ir.AttributeType.GRAPH or attr.type == ir.AttributeType.GRAPHS
            for attr in attributes.values()
        ):
            raise ValueError(
                "Inliner does not support graph attribute parameters to functions"
            )

        if len(node.inputs) > len(function.inputs):
            raise ValueError(f"Input mismatch: {len(node.inputs)} > {len(function.inputs)}")
        value_map = {}
        for i, input in enumerate(node.inputs):
            value_map[function.inputs[i]] = input
        for i in range(len(node.inputs), len(function.inputs)):
            value_map[function.inputs[i]] = None

        # Identify call-stack for node, used to generate unique names.
        call_stack = self.node_context.get(node, [])
        new_call_stack = [*call_stack, call_site_id]

        cloner = _CopyReplace(self, attributes, value_map, node.metadata_props, new_call_stack)

        # iterate over the nodes in the function, creating a copy of each node
        # and replacing inputs with the corresponding values in the value map.
        # Update the value map with the new values.

        nodes = [cloner.clone_node(node) for node in function]
        output_values = [value_map[output] for output in function.outputs]
        return nodes, output_values  # type: ignore

    def inline_calls_in(self, graph: ir.Graph) -> None:
        for input in graph.inputs:
            if input.name is not None:
                self.used_value_names.add(input.name)
        for initializer in graph.initializers:
            self.used_value_names.add(initializer)

        # Pre-processing:
        # * Count the number of times each function is called in the graph.
        #   This is used for disambiguating names of values in the inlined functions.
        # * And identify names of values that are used in the graph.
        id_count: dict[ir.OperatorIdentifier, int] = defaultdict(int)
        for node in graph:
            if node.name:
                self.used_node_names.add(node.name)
            id = node.op_identifier()
            if id in self._functions:
                id_count[id] += 1
            for output in node.outputs:
                if output.name is not None:
                    self.used_value_names.add(output.name)
        next_id: dict[ir.OperatorIdentifier, int] = defaultdict(int)
        for node in graph:
            id = node.op_identifier()
            if id in self._functions:
                # If there are multiple calls to same function, we use a prefix to disambiguate
                # the different call-sites:
                if id_count[id] > 1:
                    call_site_prefix = f"_{next_id[id]}"
                    next_id[id] += 1
                else:
                    call_site_prefix = ""
                call_site = node.name or (
                    self._function_id_abbreviations[id] + call_site_prefix
                )
                nodes, values = self._instantiate_call(node, call_site)
                ir_convenience.replace_nodes_and_values(
                    graph,
                    insertion_point=node,
                    old_nodes=[node],
                    new_nodes=nodes,
                    old_values=node.outputs,
                    new_values=values,
                )
            else:
                for attr in node.attributes.values():
                    if not isinstance(attr, ir.Attr):
                        continue
                    if attr.type == ir.AttributeType.GRAPH:
                        self.inline_calls_in(attr.as_graph())
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for graph in attr.as_graphs():
                            self.inline_calls_in(graph)


def inline(model: ir.Model) -> None:
    """Inline all function calls (recursively) in the model."""
    if model.functions:
        inliner = _Inliner(model)
        inliner.inline_calls_in(model.graph)
        model.functions.clear()
