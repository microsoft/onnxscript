# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable, Mapping, Sequence
import onnx_ir as ir

class _CopyReplace:
    """Utilities for creating a copy of IR objects with substitutions for attributes/input values."""

    def __init__(
        self,
        attr_map: Mapping[str, ir.Attr],
        value_map: dict[ir.Value, ir.Value | None],
        metadata_props: dict[str, str],
        post_process: Callable[[ir.Node], None],
    ) -> None:
        self._value_map = value_map
        self._attr_map = attr_map
        self._metadata_props = metadata_props
        self._post_process = post_process

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

    def clone_attr(self, key: str, attr: ir.Attr) -> ir.Attr | None:
        if not attr.is_ref():
            if attr.type == ir.AttributeType.GRAPH:
                graph = self.clone_graph(attr.as_graph())
                return ir.Attr(key, ir.AttributeType.GRAPH, graph, doc_string=attr.doc_string)
            elif attr.type == ir.AttributeType.GRAPHS:
                graphs = [self.clone_graph(graph) for graph in attr.as_graphs()]
                return ir.Attr(
                    key, ir.AttributeType.GRAPHS, graphs, doc_string=attr.doc_string
                )
            return attr
        assert attr.is_ref()
        ref_attr_name = attr.ref_attr_name
        assert ref_attr_name is not None, "Reference attribute must have a name"
        if ref_attr_name in self._attr_map:
            ref_attr = self._attr_map[ref_attr_name]
            if not ref_attr.is_ref():
                return ir.Attr(
                    key, ref_attr.type, ref_attr.value, doc_string=ref_attr.doc_string
                )
            assert ref_attr.ref_attr_name is not None
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
            name=node.name,
            doc_string=node.doc_string,  # type: ignore
            metadata_props=new_metadata,
        )
        new_outputs = new_node.outputs
        for i, output in enumerate(node.outputs):
            self._value_map[output] = new_outputs[i]
            new_outputs[i].name = output.name if output.name is not None else f"output_{i}"

        self._post_process(new_node)
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

def instantiate (
    function: ir.Function,
    inputs: Sequence[ir.Value | None],
    attributes: Mapping[str, ir.Attr],
    *,
    prefix: str = ""
) -> tuple[list[ir.Node], list[ir.Value | None]]:
    formal_inputs = function.inputs
    if len(inputs) > len(formal_inputs):
        raise ValueError("")
    value_map = {
        formal: actual for (formal, actual) in zip(formal_inputs, inputs)
    }
    def rename(node: ir.Node):
        if prefix != "":
            node.name = prefix + node.name
            for output in node.outputs:
                if output is not None:
                    output.name = prefix + output.name
    cloner = _CopyReplace(attributes, value_map, {}, post_process=rename)
    nodes = [cloner.clone_node(n) for n in function]
    outputs = [value_map.get(v) for v in function.outputs]
    return nodes, outputs

