"""Utilities for traversing the IR graph."""

from __future__ import annotations

__all__ = [
    "RecursiveGraphIterator",
]

from typing import Callable, Iterator, Reversible

from typing_extensions import Self

from onnxscript.ir import _core, _enums


class RecursiveGraphIterator(Iterator[_core.Node], Reversible[_core.Node]):
    def __init__(
        self,
        graph: _core.Graph | _core.Function | _core.GraphView,
        *,
        enter_graph_handler: Callable[[_core.Graph | _core.Function | _core.GraphView], None]
        | None = None,
        exit_graph_handler: Callable[[_core.Graph | _core.Function | _core.GraphView], None]
        | None = None,
        recursive: Callable[[_core.Node], bool] | None = None,
        reverse: bool = False,
    ):
        """Iterate over the nodes in the graph, recursively visiting subgraphs.

        Args:
            graph: The graph to traverse.
            enter_graph_handler: A callback that is called when a subgraph is entered.
            exit_graph_handler: A callback that is called when a subgraph is exited.
            recursive: A callback that determines whether to recursively visit a node. If
                not provided, all nodes are visited.
            reverse: Whether to iterate in reverse order.
        """
        self._graph = graph
        self._enter_graph_handler = enter_graph_handler
        self._exit_graph_handler = exit_graph_handler
        self._recursive = recursive
        self._reverse = reverse
        self._iterator = self._recursive_node_iter(graph)

    def __iter__(self) -> Self:
        self._iterator = self._recursive_node_iter(self._graph)
        return self

    def __next__(self) -> _core.Node:
        return next(self._iterator)

    def _recursive_node_iter(
        self, graph: _core.Graph | _core.Function | _core.GraphView
    ) -> Iterator[_core.Node]:
        if self._enter_graph_handler is not None:
            self._enter_graph_handler(graph)
        iterable = reversed(graph) if self._reverse else graph
        for node in iterable:  # type: ignore[union-attr]
            yield node
            if self._recursive is not None and not self._recursive(node):
                continue
            yield from self._iterate_subgraphs(node)
        if self._exit_graph_handler is not None:
            self._exit_graph_handler(graph)

    def _iterate_subgraphs(self, node: _core.Node):
        iterator = (
            reversed(node.attributes.values()) if self._reverse else node.attributes.values()
        )
        for attr in iterator:
            if not isinstance(attr, _core.Attr):
                continue
            if attr.type == _enums.AttributeType.GRAPH:
                yield from RecursiveGraphIterator(
                    attr.value,
                    enter_graph_handler=self._enter_graph_handler,
                    exit_graph_handler=self._exit_graph_handler,
                    recursive=self._recursive,
                    reverse=self._reverse,
                )
            elif attr.type == _enums.AttributeType.GRAPHS:
                graphs = reversed(attr.value) if self._reverse else attr.value
                for graph in graphs:
                    yield from RecursiveGraphIterator(
                        graph,
                        enter_graph_handler=self._enter_graph_handler,
                        exit_graph_handler=self._exit_graph_handler,
                        recursive=self._recursive,
                        reverse=self._reverse,
                    )

    def __reversed__(self) -> Iterator[_core.Node]:
        return RecursiveGraphIterator(
            self._graph,
            enter_graph_handler=self._enter_graph_handler,
            exit_graph_handler=self._exit_graph_handler,
            recursive=self._recursive,
            reverse=not self._reverse,
        )
