# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Utilities for traversing the IR graph."""

from __future__ import annotations

__all__ = [
    "RecursiveGraphIterator",
]

from typing import Callable, Iterator, Reversible, Union

from typing_extensions import Self

from onnxscript.ir import _core, _enums

GraphLike = Union[_core.Graph, _core.Function, _core.GraphView]


class RecursiveGraphIterator(Iterator[_core.Node], Reversible[_core.Node]):
    def __init__(
        self,
        graph_like: GraphLike,
        *,
        recursive: Callable[[_core.Node], bool] | None = None,
        reverse: bool = False,
    ):
        """Iterate over the nodes in the graph, recursively visiting subgraphs.

        Args:
            graph_like: The graph to traverse.
            recursive: A callback that determines whether to recursively visit the subgraphs
                contained in a node. If not provided, all nodes in subgraphs are visited.
            reverse: Whether to iterate in reverse order.
        """
        self._graph = graph_like
        self._recursive = recursive
        self._reverse = reverse
        self._iterator = self._recursive_node_iter(graph_like)

    def __iter__(self) -> Self:
        self._iterator = self._recursive_node_iter(self._graph)
        return self

    def __next__(self) -> _core.Node:
        return next(self._iterator)

    def _recursive_node_iter(
        self, graph: _core.Graph | _core.Function | _core.GraphView
    ) -> Iterator[_core.Node]:
        iterable = reversed(graph) if self._reverse else graph
        for node in iterable:  # type: ignore[union-attr]
            yield node
            if self._recursive is not None and not self._recursive(node):
                continue
            yield from self._iterate_subgraphs(node)

    def _iterate_subgraphs(self, node: _core.Node):
        for attr in node.attributes.values():
            if not isinstance(attr, _core.Attr):
                continue
            if attr.type == _enums.AttributeType.GRAPH:
                yield from RecursiveGraphIterator(
                    attr.value,
                    recursive=self._recursive,
                    reverse=self._reverse,
                )
            elif attr.type == _enums.AttributeType.GRAPHS:
                graphs = reversed(attr.value) if self._reverse else attr.value
                for graph in graphs:
                    yield from RecursiveGraphIterator(
                        graph,
                        recursive=self._recursive,
                        reverse=self._reverse,
                    )

    def __reversed__(self) -> Iterator[_core.Node]:
        return RecursiveGraphIterator(
            self._graph,
            recursive=self._recursive,
            reverse=not self._reverse,
        )
