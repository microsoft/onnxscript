# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Auxiliary class for managing names in the IR."""

from __future__ import annotations

from onnxscript.ir import _core


class NameAuthority:
    """Class for giving names to values and nodes in the IR.

    The names are generated in the format ``val_{value_counter}`` for values and
    ``node_{op_type}_{node_counter}`` for nodes. The counter is incremented each time
    a new value or node is named.

    This class keeps tracks of the names it has generated and existing names
    in the graph to prevent producing duplicated names.

    .. note::
        Once a name is tracked, it will not be made available even if the node/value
        is removed from the graph. It is possible to improve this behavior by keeping
        track of the names that are no longer used, but it is not implemented yet.

    However, if a value/node is already named when added to the graph,
    the name authority will not change its name.
    It is the responsibility of the user to ensure that the names are unique
    (typically by running a name-fixing pass on the graph).

    TODO(justichuby): Describe the pass when we have a reference implementation.
    """

    def __init__(self):
        self._value_counter = 0
        self._node_counter = 0
        self._value_names: set[str] = set()
        self._node_names: set[str] = set()

    def _unique_value_name(self) -> str:
        """Generate a unique name for a value."""
        while True:
            name = f"val_{self._value_counter}"
            self._value_counter += 1
            if name not in self._value_names:
                return name

    def _unique_node_name(self, op_type: str) -> str:
        """Generate a unique name for a node."""
        while True:
            name = f"node_{op_type}_{self._node_counter}"
            self._node_counter += 1
            if name not in self._node_names:
                return name

    def register_or_name_value(self, value: _core.Value) -> None:
        # TODO(justinchuby): Record names of the initializers and graph inputs
        if value.name is None:
            value.name = self._unique_value_name()
        # If the name is already specified, we do not change it because keeping
        # track of the used names can be costly when nodes can be removed from the graph:
        # How do we know if a name is no longer used? We cannot reserve unused names
        # because users may want to use them.
        self._value_names.add(value.name)

    def register_or_name_node(self, node: _core.Node) -> None:
        if node.name is None:
            node.name = self._unique_node_name(node.op_type)
        # If the name is already specified, we do not change it because keeping
        # track of the used names can be costly when nodes can be removed from the graph:
        # How do we know if a name is no longer used? We cannot reserve unused names
        # because users may want to use them.
        self._node_names.add(node.name)
