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
            _remove_node_and_replace_values(
                graph,
                remove_node=node,
                remove_values=node.outputs,
                new_values=existing_node.outputs,
            )
            logger.debug("Reusing node %s", existing_node)
        else:
            # If it is not, add to the mapping.
            existing_node_info_to_the_node[node_info] = node
    return modified


def _remove_node_and_replace_values(
    graph: ir.Graph,
    /,
    remove_node: ir.Node,
    remove_values: Sequence[ir.Value],
    new_values: Sequence[ir.Value],
) -> None:
    """Replaces nodes and values in the graph or function.

    Args:
        graph: The graph to replace nodes and values in.
        remove_node: The node to remove.
        remove_values: The values to replace.
        new_values: The values to replace with.
    """
    # Reconnect the users of the deleted values to use the new values
    ir.convenience.replace_all_uses_with(remove_values, new_values)
    # Update graph/function outputs if the node generates output
    if any(remove_value.is_graph_output() for remove_value in remove_values):
        replacement_mapping = dict(zip(remove_values, new_values))
        for idx, graph_output in enumerate(graph.outputs):
            if graph_output in replacement_mapping:
                new_value = replacement_mapping[graph_output]
                if new_value.is_graph_output():
                    # If the new value is also a graph output, we need to
                    # create a Identity node to preserve the remove_value.
                    identity_node = ir.node(
                        "Identity",
                        inputs=[new_value],
                        outputs=[
                            ir.Value(
                                name=graph_output.name,
                                type=graph_output.type,
                                shape=graph_output.shape,
                            )
                        ],
                    )
                    # reuse the name of the graph output
                    graph.outputs[idx] = identity_node.outputs[0]
                    graph.insert_before(
                        remove_node,
                        identity_node,
                    )
                else:
                    # if new_value is not graph output, we just
                    # update it to use old_value name.
                    new_value.name = graph_output.name
                    graph.outputs[idx] = new_value

    graph.remove(remove_node, safe=True)
