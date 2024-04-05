"""Mutable list for nodes in a graph with safe mutation properties."""

# Inspired by https://github.com/pytorch/pytorch/blob/064a650b635e6fdaa8cf1a0dbc7dbbd23a37265d/torch/fx/graph.py

from __future__ import annotations

import warnings
from typing import Iterator, Protocol, Sequence, TypeVar


class Linkable(Protocol):
    """A link in a doubly linked list that has a reference to the actual object in the link."""

    _prev: Linkable
    _next: Linkable
    _erased = False


TLinkable = TypeVar("TLinkable", bound=Linkable)


def _connect_linkables(prev: Linkable, next: Linkable) -> None:
    """Connect two linkable values."""
    prev._next = next
    next._prev = prev


def _connect_linkable_sequence(values: Sequence[Linkable]) -> None:
    """Connect a sequence of linkable values."""
    if len(values) <= 1:
        # Nothing to connect
        return
    for i in range(len(values) - 1):
        _connect_linkables(values[i], values[i + 1])


def _remove_from_list(elem: Linkable) -> None:
    """Remove a Linkable object from a doubly linked list."""
    prev, next_ = elem._prev, elem._next
    prev._next, next_._prev = next_, prev


class DoublyLinkedList(Sequence[TLinkable]):
    """A doubly linked list of nodes.

    This list supports adding and removing nodes from the list during iteration.
    """

    # TODO(justinchuby): Make it a MutableSequence

    def __init__(self, root: TLinkable) -> None:
        self._root: TLinkable = root
        self._length = 0
        self._elements = set()

    def __iter__(self) -> Iterator[TLinkable]:
        """Iterate over the elements in the list.

        - If new elements are inserted after the current node, we will
            iterate over them as well.
        - If new elements are inserted before the current node, they will
            not be iterated over in this iteration.
        - If the current node is lifted and inserted in a different location,
            iteration will start from the "next" node at the new location.
        """
        elem = self._root._next
        while elem is not self._root:
            if not elem._erased:
                yield elem
            elem = elem._next

    def __reversed__(self) -> Iterator[TLinkable]:
        """Iterate over the elements in the list in reverse order."""
        elem = self._root._prev
        while elem is not self._root:
            if not elem._erased:
                yield elem
            elem = elem._prev

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> TLinkable:
        """Get the node at the given index.

        Complexity is O(n).
        """
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

    def append(self, value: TLinkable) -> None:
        """Append a node to the list."""
        self.insert_after(self._root._prev, (value,))

    def extend(self, values: Sequence[TLinkable]) -> None:
        self.insert_after(self._root._prev, values)

    def remove(self, value: TLinkable) -> None:
        """Remove a node from the list."""
        if value._erased:
            warnings.warn(f"Element {value!r} is already erased", stacklevel=1)
            return
        if value not in self._elements:
            raise ValueError(f"Element {value!r} is not in the list")
        _remove_from_list(value)
        self._elements.remove(value)
        value._erased = True
        self._length -= 1

    def insert_after(self, value: TLinkable, new_values: Sequence[TLinkable]) -> None:
        """Insert new nodes after the given node."""
        if len(new_values) == 0:
            return
        # Create a doubly linked list of new nodes by establishing the next and prev pointers
        _connect_linkable_sequence(new_values)

        next_value = value._next
        # Insert the new nodes between the node and the next node
        _connect_linkables(value, new_values[0])
        _connect_linkables(new_values[-1], next_value)

        for v in new_values:
            # Bring the node back in case it was erased
            v._erased = False

        self._length += len(new_values)

    def insert_before(self, value: TLinkable, new_values: Sequence[TLinkable]) -> None:
        """Insert new nodes before the given node."""
        if len(new_values) == 0:
            return
        # Create a doubly linked list of new nodes by establishing the next and prev pointers
        _connect_linkable_sequence(new_values)
        prev_node = value._prev

        # Insert the new nodes between the prev node and the node
        _connect_linkables(prev_node, new_values[0])
        _connect_linkables(new_values[-1], value)

        for v in new_values:
            # Bring the node back in case it was erased
            v._erased = False

        self._length += len(new_values)
