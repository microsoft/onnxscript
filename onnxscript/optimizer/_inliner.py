# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Implementation of an inliner for onnxscript.ir"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import onnxscript.ir as ir
import onnxscript.ir.convenience as convenience

# A replacement for a node specifies a list of nodes that replaces the original node,
# and a list of values that replaces the original node's outputs.
# If the replacement is None, it indicates that the node should not be replaced.

NodeReplacement = Optional[Tuple[Iterable[ir.Node], Iterable[ir.Value]]]


class CopyReplace:
    """Utilities for creating a copy of IR objects with substitutions for attributes/input values."""

    def __init__(
        self, attr_map: dict[str, ir.Attr | ir.RefAttr], value_map: dict[ir.Value, ir.Value]
    ) -> None:
        self._value_map = value_map
        self._attr_map = attr_map

    def clone_value(self, value: ir.Value) -> ir.Value:
        # Only input-values are cloned.
        if value.producer() is not None:
            raise ValueError("Only input values should be cloned.")
        new_value = ir.Value(
            name=value.name, type=value.type, shape=value.shape, doc_string=value.doc_string
        )
        self._value_map[value] = new_value
        return new_value

    def clone_attr(self, key: str, attr: ir.Attr | ir.RefAttr) -> ir.Attr | ir.RefAttr | None:
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.GRAPH:
                return self.clone_graph(attr.value)
            elif attr.type == ir.AttributeType.GRAPHS:
                return [self.clone_graph(graph) for graph in attr.value]
            return attr
        assert isinstance(attr, ir.RefAttr)
        if key in self._attr_map:
            return self._attr_map[key]
        # Note that if a function has an attribute-parameter X, and a call (node) to the function
        # has no attribute X, all references to X in nodes inside the function body will be
        # removed. This is just the ONNX representation of optional-attributes.
        return None

    def clone_node(self, copied: ir.Node) -> ir.Node:
        new_inputs = [self._value_map[input] for input in copied.inputs]
        new_attributes = {
            key: new_value
            for key, value in copied.attributes.items()
            if (new_value := self.clone_attr(key, value)) is not None
        }
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
        return new_node

    def clone_graph(self, graph: ir.Graph) -> ir.Graph:
        input_values = [self.clone_value(v) for v in graph.inputs]
        nodes = [self.clone_node(node) for node in graph]
        initializers = graph.initializers  # Initializers are not cloned, but shared.

        return ir.Graph(
            input_values,
            graph.outputs,
            nodes=nodes,
            initializers=initializers,
            doc_string=graph.doc_string,
            opset_imports=graph.opset_imports,
            name=graph.name,
            metadata_props=graph.metadata_props,
        )


class Inliner:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self._opset_imports = model.opset_imports

    def transform_node(self, node: ir.Node) -> NodeReplacement:
        id = node.op_identifier()
        if id not in self._functions:
            return None
        # Inline the function-call

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

        cloner = CopyReplace(attributes, value_map)

        # iterate over the nodes in the function, creating a copy of each node
        # and replacing inputs with the corresponding values in the value map.
        # Update the value map with the new values.

        nodes = [cloner.clone_node(node) for node in function]
        output_values = [value_map[output] for output in function.outputs]
        return nodes, output_values

    def transform_graph(self, graph: ir.Graph | ir.Function | ir.GraphView) -> None:
        for node in graph:
            replacement = self.transform_node(node)
            if replacement is None:
                for attr in node.attributes.values():
                    if not isinstance(attr, ir.Attr):
                        continue
                    if attr.type == ir.AttributeType.GRAPH:
                        self.transform_graph(attr.value)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for graph in attr.value:
                            self.transform_graph(graph)
            else:
                nodes, values = replacement
                convenience.replace_nodes_and_values(
                    graph, node, [node], nodes, node.outputs, values
                )


def inline(model: ir.Model) -> None:
    inliner = Inliner(model)
    inliner.transform_graph(model.graph)
    model.functions.clear()
