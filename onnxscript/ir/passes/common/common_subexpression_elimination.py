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

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Return the same ir.Model but with CSE applied to the graph."""
        modified = False
        graph = model.graph

        modified = _common_subexpression_elimination(graph, modified)

        return ir.passes.PassResult(
            model,
            modified=modified,
        )


def _common_subexpression_elimination(graph: ir.Graph, modified: bool) -> bool:
    """Eliminate common subexpression in ONNX graphs."""

    # node to node identifier, length of outputs, inputs, and attributes
    existing_node_info_to_the_node: dict[
        tuple[
            ir.OperatorIdentifier,
            int,  # len(outputs)
            tuple[int, ...],  # input ids
            tuple[tuple[str, object], ...],  # attributes
        ],
        ir.Node,
    ] = {}
    previous_node = None

    for node in graph:
        # Use equality to check if the node is a common subexpression.
        attributes = {}
        for k, v in node.attributes.items():
            # TODO(exporter team): CSE subgraphs.
            # NOTE: control flow ops like Loop and If won't be CSEd
            # because attribute: graph won't match.
            attributes[k] = v.value

        node_info = (
            node.op_identifier(),
            len(node.outputs),
            tuple(id(input) for input in node.inputs),
            tuple(sorted(attributes.items())),
        )
        # Check if the node is a common subexpression.
        if node_info in existing_node_info_to_the_node:
            # If it is, this node is already in the new graph, so
            # we don't need to create a new node.
            modified = True
            existing_node = existing_node_info_to_the_node[node_info]
            ir.convenience.replace_nodes_and_values(
                graph,
                insertion_point=previous_node or node,
                old_nodes=[node],
                new_nodes=[existing_node],
                old_values=node.outputs,
                new_values=existing_node.outputs,
            )
            previous_node = existing_node
            logger.debug("Reusing node %s", existing_node.name)
        else:
            # If it is not, add to the mapping.
            existing_node_info_to_the_node[node_info] = node
            previous_node = node
    return modified
