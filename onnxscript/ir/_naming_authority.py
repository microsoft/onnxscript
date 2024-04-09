"""Module responsible for managing the names in a graph."""


class NamingAuthority:
    """Class responsible for managing the names in a graph.

    The NamingAuthority class is responsible for managing the names in a graph.

    It can do the following:
    1. Generate a unique name for an anonymous node/value.
    2. Set the name of a node/value, renaming it if the name is not unique.
    3. Set the name of a node/value, renaming conflicting nodes/values if the name is being used.

    Default naming scheme:
        For nodes: "node_{node_count}_{op_type}"
        For values: "val_{value_count}"
    """

    def __init__(self):
        self._node_names = set()
        self._value_names = set()
        # Monotonically increasing counters not affected by removal of nodes or values
        self._node_counter = 0
        self._value_counter = 0

    def name_node(self, op_type: str) -> str:
        """Generate a name for a node.

        Args:
            op_type: The type of the node.

        Returns:
            str: The name of the node.
        """
        name = f"node_{self._node_counter}_{op_type}"
        self._node_counter += 1
        self._node_names.add(name)
        return name

    def remove_node(self, name: str) -> None:
        """Remove a node name.

        Args:
            name: The name of the node.
        """
        self._node_names.remove(name)

    def name_value(self) -> str:
        """Generate a name for a value.

        Returns:
            str: The name of the value.
        """
        name = f"val_{self._value_counter}"
        self._value_counter += 1
        self._value_names.add(name)
        return name

    def remove_value(self, name: str) -> None:
        """Remove a value name.

        Args:
            name: The name of the value.
        """
        self._value_names.remove(name)
