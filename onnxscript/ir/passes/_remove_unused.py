# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Utilities for removing unused nodes the IR graph."""

from __future__ import annotations

from collections import deque
from typing import Union

import onnxscript.ir as ir
from onnxscript.ir import Graph, Node, Value


class RemoveUnused:
    def __init__(self, graph_like: Union[Graph, ir.GraphView]):
        self._graph = graph_like

    def purge(self) -> None:
        """Remove unused nodes in this graph and all its subgraphs that do not contribute to any graph_outputs."""
        # 1. Initialize
        #   Gather all nodes from the graph and its subgraphs using a recursive iterator.
        #   Identify all subgraphs by checking the graph of each node.
        #   Initialize sets to keep track of visited values and nodes.
        # 2. BFS traversal:
        #   Create a queue initialized with all output values from the subgraphs.
        #   While there are values in the queue:
        #     - Dequeue a value and retrieve its producer node.
        #     - Skip processing if the producer node is already visited or doesn't exist.
        #     - Mark the producer node as visited.
        #     - Enqueue all input values of this producer node for further exploration, if they haven't been visited.
        # 3. Remove:
        #   Remove any node from its graph if it has not been marked as visited during the BFS traversal.

        # Initialize
        all_nodes: list[Node] = list(ir.traversal.RecursiveGraphIterator(self._graph))
        subgraphs: set[Graph] = {node.graph for node in all_nodes if node.graph}
        visited_values: set[Value] = set()
        visited_nodes: set[Node] = set()

        # BFS Traversal
        value_queue: deque[Value] = deque(
            output for graph in subgraphs for output in graph.outputs if output
        )
        while value_queue:
            current_value = value_queue.popleft()
            producer_node = current_value.producer()
            if not producer_node or producer_node in visited_nodes:
                continue
            visited_nodes.add(producer_node)
            for input_value in producer_node.inputs:
                if input_value and input_value not in visited_values:
                    visited_values.add(input_value)
                    value_queue.append(input_value)

        # Remove
        for node in all_nodes:
            if node not in visited_nodes:
                node.graph.remove(node)  # type: ignore[union-attr]
