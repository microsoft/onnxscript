"""Auxiliary class for managing names in the IR."""

from __future__ import annotations

from onnxscript.ir import _core


class NameAuthority:
    """Class for giving names to values and nodes in the IR.

    The names are generated in the format ``val_{value_counter}`` for values and
    ``node_{op_type}_{node_counter}`` for nodes. The counter is incremented each time
    a new value or node is named.

    The class does not keep track of the names it has given, so it is possible to
    generate names that conflicts with existing names. It is the responsibility of the
    user to ensure that the names are unique (typically by running a name-fixing pass
    on the graph).
    """

    def __init__(self):
        self._value_counter = 0
        self._node_counter = 0

    def name_value(self, value: _core.Value) -> None:
        value.name = f"val_{self._value_counter}"
        self._value_counter += 1

    def name_node(self, node: _core.Node) -> None:
        node.name = f"node_{node.op_type}_{self._node_counter}"
        self._node_counter += 1
