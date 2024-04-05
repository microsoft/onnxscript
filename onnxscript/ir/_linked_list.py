"""Mutable list for nodes in a graph with safe mutation properties."""

# Inspired by https://github.com/pytorch/pytorch/blob/064a650b635e6fdaa8cf1a0dbc7dbbd23a37265d/torch/fx/graph.py

from __future__ import annotations


from typing import Generic, Iterator, Protocol, Sequence, TypeVar
import warnings







def _connect_nodes(prev: Node | None, next: Node | None) -> None:
    """Connect two nodes in a graph."""
    if prev is not None:
        prev._next = next
    if next is not None:
        next._prev = prev


def _connect_node_sequence(nodes: Sequence[Node]) -> None:
    """Connect a sequence of nodes in a graph."""
    if not nodes:
        return
    for i in range(len(nodes) - 1):
        _connect_nodes(nodes[i], nodes[i + 1])


class _Linkable(Protocol):
    """A class that can be attached to a linked box."""

    __link_box: LinkedBox | None = None


TLinkable = TypeVar("T", bound=_Linkable)

class LinkedBox(Generic[TLinkable]):
    """A link in a doubly linked list that has a reference to the actual object in the link."""

    def __init__(self, owning_list: DoublyLinkedList, value: TLinkable) -> None:
        self.owning_list = owning_list
        self._value = value
        self._prev = None
        self._next = None
        self._erased = False
        if value.__link_box is not None:
            raise ValueError(
                f"Node {value!r} already belongs to a linked box '{value.__link_box!r}'. "
                "Erase it from the list before adding it to another list."
            )
        value.__link_box = self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"

    def __str__(self) -> str:
        return repr(self)

    @property
    def value(self) -> TLinkable:
        return self._value

    @property
    def prev(self) -> TLinkable | None:
        return self._prev

    @property
    def next(self) -> TLinkable | None:
        return self._next

    @property
    def erased(self) -> bool:
        """Return whether the element is erased."""
        return self._erased

    def erase(self) -> None:
        """Remove the element from the list and detach the value from the linked box.

        Invariants:
            Ensures: self.erased is True
            Ensures: self.value.__link_box is None
        """
        if self._erased:
            warnings.warn(f"Node {self} is already erased", stacklevel=1)
            return
        self._erased = True
        self._value.__link_box = None

    # def __init__(self, graph: 'Graph', direction: str = '_next'):
    #     assert direction in ['_next', '_prev']
    #     self.graph = graph
    #     self.direction = direction

    # def __len__(self):
    #     return self.graph._len

    # def __iter__(self):
    #     root = self.graph._root
    #     if self.direction == "_next":
    #         cur = root._next
    #         while cur is not root:
    #             if not cur._erased:
    #                 yield cur
    #             cur = cur._next
    #     else:
    #         assert self.direction == "_prev"
    #         cur = root._prev
    #         while cur is not root:
    #             if not cur._erased:
    #                 yield cur
    #             cur = cur._prev

    # def __reversed__(self):
    #     return _node_list(self.graph, '_next' if self.direction == '_prev' else '_prev')

class DoublyLinkedList(Sequence[TLinkable]):
    """A doubly linked list of nodes.

    This list supports adding and removing nodes from the list during iteration.
    """

    def __init__(self) -> None:
        self._head: LinkedBox | None = None
        self._tail: LinkedBox | None = None
        self._length = 0

    def __iter__(self) -> Iterator[TLinkable]:
        """Iterate over the elements in the list.

        - If new elements are inserted after the current node, we will
            iterate over them as well.
        - If new elements are inserted before the current node, they will
            not be iterated over in this iteration.
        - If the current node is lifted and inserted in a different location,
            iteration will start from the "next" node at the new location.
        """
        elem = self._head
        while elem is not None:
            if not elem.is_erased():
                yield elem.value
            elem = elem.next
        # TODO: Find the right time to call _remove_erased_nodes

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> TLinkable:
        if index >= len(self):
            # TODO: check negative index too
            raise IndexError("Index out of range")
        if index < 0:
            # Look up from the end of the list
            raise NotImplementedError("Implement iteration from the back")

        iterator = iter(self)
        item = next(iterator)
        for _ in range(index):
            item = next(iterator)
        return item

    # def _remove_erased_nodes(self) -> None:
    #     node = self._head
    #     while node is not None:
    #         if node._erased:
    #             # prev <-> node <-> next
    #             # prev <-> next
    #             if node._prev is not None:
    #                 node._prev._next = node._next
    #             if node._next is not None:
    #                 node._next._prev = node._prev
    #             if self._head is node:
    #                 self._head = node._next
    #             if self._tail is node:
    #                 self._tail = node._prev
    #             node._graph = None
    #         node = node._next

    def append(self, value: TLinkable) -> None:
        """Append a node to the list."""
        # Remove the value from its original list of it is in a list
        if value.__link_box is not None:
            value.__link_box.erase()
        if len(self) == 0:
            assert self._head is None, "Bug: The head should be None when the length is 0"
            assert self._tail is None, "Bug: The tail should be None when the head is None"
            link = LinkedBox(self, value)
            self._head = link
            self._tail = link
            self._length += 1
        else:
            assert self._head is not None
            assert self._tail is not None
            # Append the node to the end of the list
            self.insert_after(self._tail, (node,))

    def extend(self, nodes: Sequence[Node]) -> None:
        if len(nodes) == 0:
            return
        if len(self) == 0:
            # Insert the first node first
            assert self._head is None, "Bug: The head should be None when the length is 0"
            assert self._tail is None, "Bug: The tail should be None when the head is None"
            first_node = nodes[0]
            first_node._erased = False
            first_node._graph = self._graph
            first_node._prev = None
            first_node._next = None
            self._head = first_node
            self._tail = first_node
            self._length += 1
        # Insert the rest of the nodes
        assert self._tail is not None
        self.insert_after(self._tail, nodes[1:])

    def erase(self, node: Node) -> None:
        """Remove a node from the list."""
        if node._erased:
            warnings.warn(f"Node {node} is already erased", stacklevel=1)
            return
        assert (
            node._graph is self._graph
        ), "Bug: Invariance violation: node is not in the graph"
        # We mark the node as erased instead of removing it from the list,
        # because removing a node from the list during iteration is not safe.
        node._erased = True
        # Remove the node from the graph
        node._graph = None
        self._length -= 1

    def insert_after(self, node: Node, new_nodes: Sequence[Node]) -> None:
        """Insert new nodes after the given node."""
        if len(new_nodes) == 0:
            return
        # Create a doubly linked list of new nodes by establishing the next and prev pointers
        _connect_node_sequence(new_nodes)
        next_node = node._next

        # Insert the new nodes between the node and the next node
        _connect_nodes(node, new_nodes[0])
        _connect_nodes(new_nodes[-1], next_node)

        # Assign graph
        for new_node in new_nodes:
            new_node._graph = self._graph
            # Bring the node back in case it was erased
            new_node._erased = False

        # Update the tail if needed
        if self._tail is node:
            # The node is the last node in the list
            self._tail = new_nodes[-1]

        self._length += len(new_nodes)

        # We don't need to update the head because any of the new nodes cannot be the head

    def insert_before(self, node: Node, new_nodes: Sequence[Node]) -> None:
        """Insert new nodes before the given node."""
        if len(new_nodes) == 0:
            return
        # Create a doubly linked list of new nodes by establishing the next and prev pointers
        _connect_node_sequence(new_nodes)
        prev_node = node._prev

        # Insert the new nodes between the prev node and the node
        _connect_nodes(prev_node, new_nodes[0])
        _connect_nodes(new_nodes[-1], node)

        # Assign graph
        for new_node in new_nodes:
            new_node._graph = self._graph
            # Bring the node back in case it was erased
            new_node._erased = False

        # Update the head if needed
        if self._head is node:
            # The node is the first node in the list
            self._head = new_nodes[0]

        self._length += len(new_nodes)

        # We don't need to update the tail because any of the new nodes cannot be the tail
