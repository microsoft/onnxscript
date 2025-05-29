# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Eliminate common subexpression in ONNX graphs."""

from __future__ import annotations

__all__ = [
    "CommonSubexpressionEliminationPass",
]

import logging
from typing import Sequence

from onnxscript import ir

logger = logging.getLogger(__name__)


class CommonSubexpressionEliminationPass(ir.passes.InPlacePass):
    """Eliminate common subexpression in ONNX graphs."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Return the same ir.Model but with CSE applied to the graph."""
        modified = False
        graph = model.graph

        modified = _eliminate_common_subexpression(graph, modified)

        return ir.passes.PassResult(
            model,
            modified=modified,
        )


def _eliminate_common_subexpression(graph: ir.Graph, modified: bool) -> bool:
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

    for node in graph:
        # Skip control flow ops like Loop and If.
        control_flow_op: bool = False
        # Use equality to check if the node is a common subexpression.
        attributes = {}
        for k, v in node.attributes.items():
            # TODO(exporter team): CSE subgraphs.
            # NOTE: control flow ops like Loop and If won't be CSEd
            # because attribute: graph won't match.
            if v.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS):
                control_flow_op = True
                logger.debug("Skipping control flow op %s", node)
            # The attribute value could be directly taken from the original
            # protobuf, so we need to make a copy of it.
            value = v.value
            if v.type in (
                ir.AttributeType.INTS,
                ir.AttributeType.FLOATS,
                ir.AttributeType.STRINGS,
            ):
                # For INT, FLOAT and STRING attributes, we convert them to tuples
                # to ensure they are hashable.
                value = tuple(value)
            attributes[k] = value

        if control_flow_op:
            # If the node is a control flow op, we skip it.
            continue

        node_info = (
            node.op_identifier(),
            len(node.outputs),
            tuple(id(input) for input in node.inputs),
            tuple(sorted(attributes.items())),
        )

        # Check if the node is a common subexpression.
        if node_info in existing_node_info_to_the_node:
            # If it is, this node has an existing node with the same
            # operator, number of outputs, inputs, and attributes.
            # We replace the node with the existing node.
            modified = True
            existing_node = existing_node_info_to_the_node[node_info]
            _remove_node_and_replace__values(
                graph,
                remove_nodes=[node],
                remove_values=node.outputs,
                new_values=existing_node.outputs,
            )
            logger.debug("Reusing node %s", existing_node)
        else:
            # If it is not, add to the mapping.
            existing_node_info_to_the_node[node_info] = node
    return modified


def _remove_node_and_replace__values(
    graph: ir.Graph,
    /,
    remove_nodes: ir.Node,
    remove_values: Sequence[ir.Value],
    new_values: Sequence[ir.Value],
) -> None:
    """Replaces nodes and values in the graph or function.

    Args:
        graph: The graph to replace nodes and values in.
        remove_nodes: The nodes to remove.
        remove_values: The values to replace.
        new_values: The values to replace with.
    """

    for old_value, new_value in zip(remove_values, new_values):
        # Propagate relevant info from old value to new value
        # TODO(Rama): Perhaps this should be a separate utility function. Also, consider
        # merging old and new type/shape info.
        new_value.type = old_value.type
        new_value.shape = old_value.shape
        new_value.const_value = old_value.const_value
        new_value.name = old_value.name

    # Reconnect the users of the deleted values to use the new values
    ir.convenience.replace_all_uses_with(remove_values, new_values)
    # Update graph/function outputs if the node generates output
    replacement_mapping = dict(zip(remove_values, new_values))
    for idx, graph_or_function_output in enumerate(graph.outputs):
        if graph_or_function_output in replacement_mapping:
            graph.outputs[idx] = replacement_mapping[graph_or_function_output]

    graph.remove(remove_nodes, safe=True)
