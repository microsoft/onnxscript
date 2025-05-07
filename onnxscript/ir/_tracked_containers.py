# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tracked containers for graph."""

from __future__ import annotations

__all__ = [
    "GraphInputs",
    "GraphOutputs",
]

import collections
import logging
from typing import TYPE_CHECKING, Iterable, SupportsIndex

import onnxscript

if TYPE_CHECKING:
    from onnxscript.ir import _core


logger = logging.getLogger(__name__)


class _GraphIO(collections.UserList[_core.Value]):
    """The inputs and outputs of a Graph."""

    def __init__(self, graph: _core.Graph, initlist=None):
        super().__init__(initlist)
        self._graph = graph

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        raise NotImplementedError

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        raise NotImplementedError

    def _unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        raise NotImplementedError

    def append(self, item: _core.Value) -> None:
        """Add a new input to the graph."""
        super().append(item)
        self._set_graph(item)
        self._check_invariance()

    def extend(self, other) -> None:
        """Extend the list of inputs or outputs."""
        super().extend(other)
        for item in other:
            self._set_graph(item)

    def insert(self, i: int, item: _core.Value) -> None:
        """Insert an input/output to the graph."""
        super().insert(i, item)
        self._set_graph(item)
        self._check_invariance()

    def pop(self, i: int = -1) -> _core.Value:
        """Remove an input/output from the graph."""
        value = super().pop(i)
        self._unset_graph(value)
        self._check_invariance()
        return value

    def remove(self, item: _core.Value) -> None:
        """Remove an input/output from the graph."""
        super().remove(item)
        self._unset_graph(item)
        self._check_invariance()

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
            self._check_invariance()
            return
        elif isinstance(item, _core.Value) and isinstance(i, SupportsIndex):
            # Replace a single item
            self._unset_graph(self.data[i])
            self._set_graph(item)
            super().__setitem__(i, item)
            self._check_invariance()
            return

        raise TypeError(f"Invalid types for __setitem__: {type(i)} and {type(item)}")


class GraphInputs(_GraphIO):
    """The inputs of a Graph."""

    def __init__(self, graph: _core.Graph, initlist=None):
        super().__init__(graph, initlist)

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnxscript.DEBUG:
            return
        for value in self.data:
            if value._graph_input_of is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an input of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph_input_of is not None and value._graph_input_of is not self._graph:
            logger.warning(
                "Value '%s' is already an input of a different graph. Overwriting",
                value,
            )
        value._graph_input_of = self._graph

    def _unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        if value._graph_input_of is not self._graph:
            # The value is already added to a different graph
            return
        value._graph_input_of = None


class GraphOutputs(_GraphIO):
    """The outputs of a Graph."""

    def __init__(self, graph: _core.Graph, initlist=None):
        super().__init__(graph, initlist)

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnxscript.DEBUG:
            return
        for value in self.data:
            if value._graph_output_of is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an output of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph_output_of is not None and value._graph_output_of is not self._graph:
            logger.warning(
                "Value '%s' is already an output of a different graph. Overwriting",
                value,
            )
        value._graph_output_of = self._graph

    def _unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        if value._graph_output_of is not self._graph:
            # The value is already added to a different graph
            return
        value._graph_output_of = None
