# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tracked containers for graph."""

# pylint: disable=protected-access

from __future__ import annotations

__all__ = [
    "GraphInputs",
    "GraphOutputs",
]

import collections
from typing import TYPE_CHECKING, Iterable, SupportsIndex

import onnxscript

if TYPE_CHECKING:
    from onnxscript.ir import _core


class _GraphIO(collections.UserList["_core.Value"]):
    """The inputs and outputs of a Graph."""

    def __init__(self, graph: _core.Graph, initlist=None):
        self._graph = graph
        # Use a ref counter to track the number of references to each value
        # in the input/output list. This is used to determine when to unset the graph
        # reference in the value.
        # Even though a duplicated value is invalid in inputs and not recommended in outputs,
        # it is still possible to have duplicated inputs/outputs in an ONNX graph so we
        # need to properly handle this case and maintain the graph reference properly.
        self._ref_counter: collections.Counter[_core.Value] = collections.Counter()
        if initlist is not None:
            initlist = tuple(initlist)  # Create a copy in case initlist is a generator
            for value in initlist:
                self._set_graph(value)
        super().__init__(initlist)
        self._check_invariance()

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        raise NotImplementedError

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        raise NotImplementedError

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        raise NotImplementedError

    def append(self, item: _core.Value) -> None:
        """Add a new input to the graph."""
        # Perform checks first in _set_graph before modifying the data structure
        self._set_graph(item)
        super().append(item)
        self._check_invariance()

    def extend(self, other) -> None:
        """Extend the list of inputs or outputs."""
        other = tuple(other)
        for item in other:
            self._set_graph(item)
        super().extend(other)

    def insert(self, i: int, item: _core.Value) -> None:
        """Insert an input/output to the graph."""
        super().insert(i, item)
        self._set_graph(item)
        self._check_invariance()

    def pop(self, i: int = -1) -> _core.Value:
        """Remove an input/output from the graph."""
        value = super().pop(i)
        self._maybe_unset_graph(value)
        self._check_invariance()
        return value

    def remove(self, item: _core.Value) -> None:
        """Remove an input/output from the graph."""
        super().remove(item)
        self._maybe_unset_graph(item)
        self._check_invariance()

    def clear(self) -> None:
        """Clear the list."""
        for value in self.data:
            self._maybe_unset_graph(value)
        super().clear()

    def __setitem__(self, i, item) -> None:
        """Replace an input/output to the node."""
        if isinstance(item, Iterable) and isinstance(i, slice):
            # Modify a slice of the list
            for value in self.data[i]:
                self._maybe_unset_graph(value)
            for value in item:
                self._set_graph(value)
            super().__setitem__(i, item)
            self._check_invariance()
            return
        elif isinstance(i, SupportsIndex):
            # Replace a single item
            self._maybe_unset_graph(self.data[i])
            self._set_graph(item)
            super().__setitem__(i, item)
            self._check_invariance()
            return

        raise TypeError(f"Invalid types for __setitem__: {type(i)} and {type(item)}")

    def __getitem__(self, i):
        """Get an input/output from the graph."""
        return self.data[i]

    def _unimplemented(self, *_args, **_kwargs):
        """Unimplemented method."""
        raise RuntimeError("Method is not supported")

    __add__ = _unimplemented
    __radd__ = _unimplemented
    __iadd__ = _unimplemented
    __mul__ = _unimplemented
    __rmul__ = _unimplemented
    copy = _unimplemented


class GraphInputs(_GraphIO):
    """The inputs of a Graph."""

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnxscript.DEBUG:
            return
        for value in self.data:
            if value._graph is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an input of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already owned by a different graph. Please remove the value from the previous graph first"
            )
        self._ref_counter[value] += 1
        value._is_graph_input = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        self._ref_counter[value] -= 1
        if self._ref_counter[value] > 0:
            # The value is still used by another graph input
            return
        value._is_graph_input = False
        if value._owned_by_graph():
            # Keep the graph reference if the value is still an input or an initializer
            return
        value._graph = None


class GraphOutputs(_GraphIO):
    """The outputs of a Graph."""

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnxscript.DEBUG:
            return
        for value in self.data:
            if value._graph is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an output of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an output of a different graph. Please remove the value from the previous graph first"
            )
        self._ref_counter[value] += 1
        value._is_graph_output = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        self._ref_counter[value] -= 1
        if self._ref_counter[value] > 0:
            # The value is still used by another graph input
            return
        value._is_graph_output = False
        if value._owned_by_graph():
            # Keep the graph reference if the value is still an input or an initializer
            return
        value._graph = None


class GraphInitializers(collections.UserDict[str, "_core.Value"]):
    """The initializers of a Graph."""

    def __init__(self, graph: _core.Graph, dict=None, /, **kwargs):
        # Perform checks first in _set_graph before modifying the data structure with super().__init__()
        data = {}
        if dict is not None:
            data.update(dict)
        if kwargs:
            data.update(kwargs)
        self._graph = graph
        for value in data.values():
            self._set_graph(value)

        super().__init__(data)

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an initializer of a different graph. Please remove the value from the previous graph first"
            )
        value._is_initializer = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        value._is_initializer = False
        if value._owned_by_graph():
            # Keep the graph reference if the value is still an input or an initializer
            return
        value._graph = None

    def __setitem__(self, key: str, value: _core.Value) -> None:
        """Set an initializer for the graph."""
        if key != value.name:
            raise ValueError(
                f"Key '{key}' does not match the name of the value '{value.name}'"
            )
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, not {type(key)}")
        if key in self.data:
            # If the key already exists, unset the old value
            old_value = self.data[key]
            self._maybe_unset_graph(old_value)
        # Must call _set_graph before super().__setitem__ so that when there is an error,
        # the dictionary is not modified
        self._set_graph(value)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete an initializer from the graph."""
        value = self.data[key]
        # Must call _maybe_unset_graph before super().__delitem__ so that when there is an error,
        # the dictionary is not modified
        self._maybe_unset_graph(value)
        super().__delitem__(key)
