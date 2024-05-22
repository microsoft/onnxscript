"""Utilities for traversing the IR graph."""

from __future__ import annotations

from typing_extensions import Self

from onnxscript.ir import _core, _enums

__all__ = [
    "RecursiveNodeIterator",
]

from typing import Callable, Iterator


class RecursiveNodeIterator(Iterator[_core.Node]):
    def __init__(
        self,
        graph: _core.Graph,
        enter_graph_handler: Callable[[_core.Graph], None],
        exit_graph_handler: Callable[[_core.Graph], None],
        recursive: Callable[[_core.Node], bool],
        reversed: bool = False,
    ):
        """Iterate over the nodes in the graph, recursively visiting subgraphs.

        Args:
            graph: The graph to traverse.
            enter_graph_handler: A callback that is called when a subgraph is entered.
            exit_graph_handler: A callback that is called when a subgraph is exited.
            recursive: A callback that determines whether to recursively visit a node.
            reversed: Whether to iterate in reverse order.
        """
        self._graph = graph
        self._enter_graph_handler = enter_graph_handler
        self._exit_graph_handler = exit_graph_handler
        self._recursive = recursive
        self._subgraph_stack = []
        self._reversed = reversed
        self._iterator = self._recursive_node_iter(graph)

    def __iter__(self) -> Self:
        self._iterator = self._recursive_node_iter(self._graph)
        return self

    def __next__(self) -> _core.Node:
        return next(self._iterator)

    def _recursive_node_iter(self, graph: _core.Graph) -> Iterator[_core.Node]:
        self._enter_graph_handler(graph)
        iterable = reversed(graph) if self._reversed else graph
        for node in iterable:
            yield node
            if not self._recursive(node):
                continue
            yield from self._iterate_subgraphs(node)
        self._exit_graph_handler(graph)

    def _iterate_subgraphs(self, node: _core.Node):
        for attr in node.attributes.values():
            if not isinstance(attr, _core.Attr):
                continue
            if attr.type == _enums.AttributeType.GRAPH:
                yield from RecursiveNodeIterator(
                    attr.value,
                    self._enter_graph_handler,
                    self._exit_graph_handler,
                    self._recursive,
                    reversed=self._reversed,
                )
            elif attr.type == _enums.AttributeType.GRAPHS:
                for graph in attr.value:
                    yield from RecursiveNodeIterator(
                        graph,
                        self._enter_graph_handler,
                        self._exit_graph_handler,
                        self._recursive,
                        reversed=self._reversed,
                    )

    def __reversed__(self) -> Iterator[_core.Node]:
        return RecursiveNodeIterator(
            self._graph,
            self._enter_graph_handler,
            self._exit_graph_handler,
            self._recursive,
            reversed=not self._reversed,
        )
