# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Implementation of an inliner for onnxscript.ir"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Tuple

import onnxscript.ir as ir
import onnxscript.ir.convenience as convenience

# A replacement for a node specifies a list of nodes that replaces the original node,
# and a list of values that replaces the original node's outputs.

NodeReplacement = Tuple[Sequence[ir.Node], Sequence[ir.Value]]

# A call stack is a list of identifiers of call sites, where the first element is the
# outermost call site, and the last element is the innermost call site.
CallSiteId = str
CallStack = list[CallSiteId]

class CopyReplace:
    """Utilities for creating a copy of IR objects with substitutions for attributes/input values."""

    def __init__(
        self,
        attr_map: dict[str, ir.Attr | ir.RefAttr],
        value_map: dict[ir.Value, ir.Value | None],
        used_value_names: set[str],
        node_context: dict[ir.Node, CallStack],
        call_stack: CallStack,
    ) -> None:
        self._value_map = value_map
        self._attr_map = attr_map
        self._node_context = node_context
        self._call_stack = call_stack
        self._used_value_names = used_value_names

    def make_unique_name(self, name: str) -> str:
        # Generate a unique name by appending a number to the name until it is unique.
        prefix = "_".join(self._call_stack)
        name = prefix + "_" + name
        candidate = name
        i = 1
        while candidate in self._used_value_names:
            i += 1
            candidate = f"{name}_{i}"
        self._used_value_names.add(candidate)
        return candidate

    def clone_value(self, value: ir.Value) -> ir.Value | None:
        # Only input-values are cloned.
        if value in self._value_map:
            return self._value_map[value]
        assert value.producer() is not None, f"Value {value} has no entry in the value map"
        new_value = ir.Value(
            name=value.name, type=value.type, shape=value.shape, doc_string=value.doc_string
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
                graph = self.clone_graph(attr.value)
                return ir.Attr(key, ir.AttributeType.GRAPH, graph, doc_string=attr.doc_string)
            elif attr.type == ir.AttributeType.GRAPHS:
                graphs = [self.clone_graph(graph) for graph in attr.value]
                return ir.Attr(
                    key, ir.AttributeType.GRAPHS, graphs, doc_string=attr.doc_string
                )
            return attr
        assert isinstance(attr, ir.RefAttr)
        if key in self._attr_map:
            return self._attr_map[key]
        # Note that if a function has an attribute-parameter X, and a call (node) to the function
        # has no attribute X, all references to X in nodes inside the function body will be
        # removed. This is just the ONNX representation of optional-attributes.
        return None

    def clone_node(self, copied: ir.Node) -> ir.Node:
        new_inputs = [self.clone_optional_value(input) for input in copied.inputs]
        new_attributes = [
            new_value
            for key, value in copied.attributes.items()
            if (new_value := self.clone_attr(key, value)) is not None
        ]
        new_node = ir.Node(
            copied.domain,
            copied.op_type,
            new_inputs,
            new_attributes,
            overload=copied.overload,
            num_outputs=len(copied.outputs),
            graph=None,  #  TODO:
            name=copied.name,  #  TODO: add a unique name
            doc_string=copied.doc_string,
            metadata_props=copied.metadata_props,
        )
        new_outputs = new_node.outputs
        for i, output in enumerate(copied.outputs):
            self._value_map[output] = new_outputs[i]
            old_name = output.name if output.name is not None else f"output_{i}"
            new_outputs[i].name = self.make_unique_name(old_name)

        self._node_context[new_node] = self._call_stack

        return new_node

    def clone_graph(self, graph: ir.Graph) -> ir.Graph:
        input_values = [self.clone_value(v) for v in graph.inputs]
        nodes = [self.clone_node(node) for node in graph]
        initializers = graph.initializers  # Initializers are not cloned, but shared.

        return ir.Graph(
            input_values,  # type: ignore
            graph.outputs,
            nodes=nodes,
            initializers=list(initializers.values()),
            doc_string=graph.doc_string,
            opset_imports=graph.opset_imports,
            name=graph.name,
            metadata_props=graph.metadata_props,
        )


class Inliner:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self._opset_imports = model.opset_imports
        self._node_context : dict[ir.Node, CallStack] = {}
        self._used_value_names = set()
        function_ids = self._functions.keys()
        def id_abbreviation(id: Tuple[str, str, str]) -> str:
            """Create a short unambiguous abbreviation for a function id."""
            domain, name, overload = id
            if any (x[0] != domain and x[1] == name and x[2] == overload for x in function_ids):
                short_domain = domain + "_"
            else:
                short_domain = ""
            if overload != "":
                return short_domain + name + "_" + overload
            return short_domain + name
        self._function_id_abbreviations = {id: id_abbreviation(id) for id in function_ids}

    def transform_node(self, node: ir.Node, call_site_id: str) -> NodeReplacement:
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
        attributes = node.attributes
        if len(node.inputs) > len(function.inputs):
            raise ValueError(f"Input mismatch: {len(node.inputs)} > {len(function.inputs)}")
        value_map = {}
        for i, input in enumerate(node.inputs):
            value_map[function.inputs[i]] = input
        for i in range(len(node.inputs), len(function.inputs)):
            value_map[function.inputs[i]] = None

        # Identify call-stack for node, used to generate unique names.
        call_stack = self._node_context.get(node, [])
        call_stack.append(call_site_id)

        cloner = CopyReplace(attributes, value_map, self._used_value_names, self._node_context, call_stack)

        # iterate over the nodes in the function, creating a copy of each node
        # and replacing inputs with the corresponding values in the value map.
        # Update the value map with the new values.

        nodes = [cloner.clone_node(node) for node in function]
        output_values = [value_map[output] for output in function.outputs]
        return nodes, output_values  # type: ignore

    def transform_graph(self, graph: ir.Graph) -> None:
        for input in graph.inputs:
            if input.name is not None:
                self._used_value_names.add(input.name)
        for initializer in graph.initializers:
            self._used_value_names.add(initializer)

        # Pre-processing:
        # * Count the number of times each function is called in the graph.
        #   This is used for disambiguating names of values in the inlined functions.
        # * And identify names of values that are used in the graph.
        id_count : dict[ir.OperatorIdentifier, int] = defaultdict(int)
        for node in graph:
            id = node.op_identifier()
            if id in self._functions:
                id_count[id] += 1
            for output in node.outputs:
                if output.name is not None:
                    self._used_value_names.add(output.name)
        next_id : dict[ir.OperatorIdentifier, int] = defaultdict(int)
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
                call_site = node.name or (self._function_id_abbreviations[id] + call_site_prefix)
                nodes, values = self.transform_node(node, call_site)
                convenience.replace_nodes_and_values(
                    graph, node, [node], nodes, node.outputs, values
                )
            else:
                for attr in node.attributes.values():
                    if not isinstance(attr, ir.Attr):
                        continue
                    if attr.type == ir.AttributeType.GRAPH:
                        self.transform_graph(attr.value)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for graph in attr.value:
                            self.transform_graph(graph)



def inline(model: ir.Model) -> None:
    inliner = Inliner(model)
    inliner.transform_graph(model.graph)
    model.functions.clear()
