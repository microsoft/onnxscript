# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Utilities for removing unused nodes the IR graph."""

from __future__ import annotations

from collections import deque

import onnxscript.ir as ir
from onnxscript.ir import Attr, Graph, Node, Value, _enums


class RemoveUnused:
    def __init__(self, graph_like: Graph):
        self._graph = graph_like

    def purge(self) -> None:
        """Remove unused nodes in this graph (and all subgraphs) that do not contribute to main graph outputs."""
        # 1. Initialize:
        #   Gather all nodes from the graph and its subgraphs.
        #   Initialize sets to keep track of visited graphs, values, and nodes.
        # 2. BFS traversal:
        #   Create a queue initialized with all output values of the main graph.
        #   While there are values in the queue:
        #     - Dequeue a value and retrieve its producer node.
        #     - Mark the producer node as visited, if it hasn't been visited.
        #     - Enqueue all output values of the attribute subgraphs of the producer node,
        #       if they haven't been visited.
        #     - Enqueue all input values of the producer node, if they haven't been visited.
        # 3. Remove:
        #   Remove all nodes that have not been marked as visited during the BFS traversal.

        # Initialize
        all_nodes: list[Node] = list(ir.traversal.RecursiveGraphIterator(self._graph))
        visited_graphs: set[Graph] = set()
        visited_values: set[Value] = set()
        visited_nodes: set[Node] = set()

        # BFS Traversal
        queue: deque[Value] = deque()

        def add_graph_output_values_to_queue(graph: Graph | None) -> None:
            """Helper function to add all output values of a graph to the queue."""
            if not graph or graph in visited_graphs:
                return
            visited_graphs.add(graph)
            for output in graph.outputs:
                if not output:
                    continue
                queue.append(output)
                visited_values.add(output)

        add_graph_output_values_to_queue(self._graph)

        while queue:
            # Dequeue a value and retrieve its producer_node
            # Add producer_node to visited_nodes
            current_value = queue.popleft()
            producer_node = current_value.producer()
            if not producer_node or producer_node in visited_nodes:
                continue
            visited_nodes.add(producer_node)
            # Add producer_node's subgraphs to visited_graphs
            # Add subgraphs' output values to queue
            for attr in producer_node.attributes.values():
                if not isinstance(attr, Attr):
                    continue
                if attr.type == _enums.AttributeType.GRAPH:
                    add_graph_output_values_to_queue(attr.value)
                elif attr.type == _enums.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        add_graph_output_values_to_queue(subgraph)
            # Add producer_node's input values to queue
            for input_value in producer_node.inputs:
                if input_value and input_value not in visited_values:
                    visited_values.add(input_value)
                    queue.append(input_value)

        # Remove
        for node in all_nodes:
            if node not in visited_nodes: # type: ignore[union-attr]`
                node.graph.remove(node)
