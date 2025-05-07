# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tracked lists for graph and node IO."""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Iterable, Literal, SupportsIndex

if TYPE_CHECKING:
    from onnxscript.ir import _core


class GraphIO(collections.UserList[_core.Value]):
    """The inputs and outputs of a Graph."""

    def __init__(self, graph: _core.Graph, typ: Literal["input", "output"], initlist=None):
        super().__init__(initlist)
        self._graph = graph
        assert typ in {"intput", "output"}
        self._typ = typ

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph_input_of is not None and value._graph_input_of is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an input of a different graph: {value._graph_input_of!r}"
            )
        if value._graph_output_of is not None and value._graph_output_of is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an output of a different graph: {value._graph_output_of!r}"
            )

        if self._typ == "input":
            value._graph_input_of = self._graph
        else:
            value._graph_output_of = self._graph

    def _unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        if self._typ == "input":
            value._graph_input_of = None
        else:
            value._graph_output_of = None

    def append(self, item: _core.Value) -> None:
        """Add a new input to the graph."""
        super().append(item)
        self._set_graph(item)

    def extend(self, other) -> None:
        """Extend the list of inputs or outputs."""
        super().extend(other)
        for item in other:
            self._set_graph(item)

    def insert(self, i: int, item: _core.Value) -> None:
        """Insert an input/output to the graph."""
        super().insert(i, item)
        self._set_graph(item)

    def pop(self, i: int = -1) -> _core.Value:
        """Remove an input/output from the graph."""
        value = super().pop(i)
        self._unset_graph(value)
        return value

    def remove(self, item: _core.Value) -> None:
        """Remove an input/output from the graph."""
        super().remove(item)
        self._unset_graph(item)

    def clear(self) -> None:
        """Clear the list."""
        for value in self.data:
            self._unset_graph(value)
        super().clear()

    def __setitem__(self, i, item) -> None:
        """Replace an input/output to the node."""
        if isinstance(item, Iterable) and isinstance(i, slice):
            # Modify a slice of the list
            for value in self.data[i]:
                self._unset_graph(value)
            for value in item:
                self._set_graph(value)
            super().__setitem__(i, item)
            return
        elif isinstance(item, _core.Value) and isinstance(i, SupportsIndex):
            # Replace a single item
            self._unset_graph(self.data[i])
            self._set_graph(item)
            super().__setitem__(i, item)
            return

        raise TypeError(f"Invalid types for __setitem__: {type(i)} and {type(item)}")
