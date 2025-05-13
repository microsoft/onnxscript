# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Eliminate common subexpression in ONNX graphs."""

from __future__ import annotations

__all__ = [
    "CommonSubexpressionEliminationPass",
]

import logging

from onnxscript import ir

logger = logging.getLogger(__name__)


class CommonSubexpressionEliminationPass(ir.passes.InPlacePass):
    """Eliminate common subexpression in ONNX graphs."""

    # TODO(titaiwang): Logger
    # TODO(titaiwang): Add more tests
    # TODO(titaiwang): FunctionalPass?
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Return the same ir.Model but with CSE applied to the graph."""
        modified = False
        # 1. Initialize a new graph. It will be used to store the new nodes.
        #    and replace the old graph.
        old_graph = model.graph
        # Values and nodes point to the old graph, so they need to be
        # created in the new graph.
        new_graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            initializers=[],
            name=old_graph.name,
            metadata_props=old_graph.metadata_props,
            opset_imports=old_graph.opset_imports,
            doc_string=old_graph.doc_string,
        )
        # 2. Create a mapping from olds to news.
        old_node_hash_to_new_node: dict[int, ir.Node] = {}
        old_value_to_new_value: dict[ir.Value, ir.Value] = {}

        # 3. Create inputs and initializers in the new graph
        #    from the old graph.
        for input in old_graph.inputs:
            new_input = _copy_value(input)
            new_graph.inputs.append(new_input)
            old_value_to_new_value[input] = new_input
        for initializer in old_graph.initializers.values():
            new_initializer = _copy_value(initializer)
            new_graph.register_initializer(new_initializer)
            old_value_to_new_value[initializer] = new_initializer

        # 4. Create nodes in the new graph from the old graph.
        for old_node in old_graph:
            # 4.0. Iterate and update node inputs.
            old_node_inputs: list[ir.Value] = []
            for old_input in old_node.inputs:
                assert old_input is not None
                if old_input in old_value_to_new_value:
                    old_node_inputs.append(old_value_to_new_value[old_input])
                else:
                    old_node_inputs.append(old_input)
            # 4.1. Construct the (node, inputs, attributes) hash to
            #      check if the node is a common subexpression.
            hash_value = hash(
                (
                    old_node.op_identifier(),
                    tuple(old_node_inputs),
                    tuple(old_node.attributes.items()),
                )
            )
            # TODO(exporter team): Subgraphs are not supported yet.
            # TODO(exporter team): Skip control flow nodes?
            # 4.2. Check if the node is a common subexpression.
            if hash_value in old_node_hash_to_new_node:
                # 4.2.1. If it is, this node is already in the new graph, so
                #         we don't need to create a new node.
                modified = True
                new_node = old_node_hash_to_new_node[hash_value]
            else:
                # 4.2.2. If it is not, create a new node and add it to the graph.
                new_node = _copy_node(old_node, old_value_to_new_value)
                new_graph.append(new_node)
                old_node_hash_to_new_node[hash_value] = new_node
            # 4.3 Add the node outputs to the mapping.
            old_value_to_new_value.update(dict(zip(old_node.outputs, new_node.outputs)))
        # 5. Create outputs in the new graph from the old graph.
        for output in old_graph.outputs:
            new_output = old_value_to_new_value[output]
            new_graph.outputs.append(new_output)
        # 6. Replace the old graph with the new graph.
        model.graph = new_graph
        return ir.passes.PassResult(
            model,
            modified=modified,
        )


def _copy_value(original_value: ir.Value) -> ir.Value:
    """Copy an IR value."""
    new_input = ir.Value(
        name=original_value.name,
        shape=original_value.shape,
        type=original_value.type,
        doc_string=original_value.doc_string,
        const_value=original_value.const_value,
    )
    return new_input


def _copy_node(
    original_node: ir.Node, old_value_to_new_value: dict[ir.Value, ir.Value]
) -> ir.Node:
    """Copy an IR node."""
    new_inputs: list[ir.Value] = []
    for original_input in original_node.inputs:
        if original_input in old_value_to_new_value:
            new_inputs.append(old_value_to_new_value[original_input])
        else:
            raise ValueError(f"Input {original_input} not found in old_value_to_new_value")
    new_node = ir.node(
        domain=original_node.domain,
        op_type=original_node.op_type,
        inputs=new_inputs,
        attributes=original_node.attributes,
        overload=original_node.overload,
        num_outputs=len(original_node.outputs),
        metadata_props=original_node.metadata_props,
        doc_string=original_node.doc_string,
        name=original_node.name,
        version=original_node.version,
    )
    return new_node
