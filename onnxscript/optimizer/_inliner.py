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


class Inliner:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self._opset_imports = model.opset_imports

    def transform_node(self, node: ir.Node) -> NodeReplacement:
        id = node.op_identifier()
        if id not in self._functions:
            return None
        function = self._functions[id]

        # check opset compatibility
        for key, value in function.opset_imports.items():
            if key not in self._opset_imports:
                self._opset_imports[key] = value
            elif self._opset_imports[key] != value:
                raise ValueError(
                    f"Opset mismatch: {key} {self._opset_imports[key]} != {value}"
                )

        attributes = node.attributes
        if len(node.inputs) > len(function.inputs):
            raise ValueError(f"Input mismatch: {len(node.inputs)} > {len(function.inputs)}")
        value_map = {}
        for i, input in enumerate(node.inputs):
            value_map[function.inputs[i]] = input
        for i in range(len(node.inputs), len(function.inputs)):
            value_map[function.inputs[i]] = None

        def substitute_attr(
            key: str, value: ir.Attr | ir.RefAttr
        ) -> ir.Attr | ir.RefAttr | None:
            if isinstance(value, ir.Attr):
                # TODO: handle subgraphs
                return value
            assert isinstance(value, ir.RefAttr)
            if key in attributes:
                return attributes[key]
            return None

        # iterate over the nodes in the function, creating a copy of each node
        # and replacing inputs with the corresponding values in the value map.
        # Update the value map with the new values.
        def copy_node(copied: ir.Node) -> ir.Node:
            new_inputs = [value_map[input] for input in copied.inputs]
            new_attributes = {
                key: new_value
                for key, value in copied.attributes.items()
                if (new_value := substitute_attr(key, value)) is not None
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
                value_map[output] = new_outputs[i]
            return new_node

        nodes = [copy_node(node) for node in function]
        output_values = [value_map[output] for output in function.outputs]
        return nodes, output_values

    def _visit_graph(self, graph: ir.Graph | ir.Function | ir.GraphView) -> None:
        for node in graph:
            replacement = self.transform_node(node)
            if replacement is None:
                for attr in node.attributes.values():
                    if not isinstance(attr, ir.Attr):
                        continue
                    if attr.type == ir.AttributeType.GRAPH:
                        self._visit_graph(attr.value)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for graph in attr.value:
                            self._visit_graph(graph)
            else:
                nodes, values = replacement
                convenience.replace_nodes_and_values(
                    graph, node, [node], nodes, node.outputs, values
                )


def inline(model: ir.Model) -> None:
    inliner = Inliner(model)
    inliner._visit_graph(model.graph)
    model.functions.clear()
