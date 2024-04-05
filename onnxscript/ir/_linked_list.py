"""Mutable list for nodes in a graph with safe mutation properties."""

# Inspired by https://github.com/pytorch/pytorch/blob/064a650b635e6fdaa8cf1a0dbc7dbbd23a37265d/torch/fx/graph.py

from __future__ import annotations

import warnings
from typing import Callable, Generic, Iterable, Iterator, Protocol, TypeVar


class Linkable(Protocol):
    """A link in a doubly linked list that has a reference to the actual object in the link."""

    _prev: Linkable
    _next: Linkable
    _erased = False


TLinkable = TypeVar("TLinkable", bound=Linkable)


def _remove_from_list(elem: Linkable) -> None:
    """Remove a Linkable object from a doubly linked list."""
    prev, next_ = elem._prev, elem._next
    prev._next, next_._prev = next_, prev


class DoublyLinkedList(Generic[TLinkable], Iterable[TLinkable]):
    """A doubly linked list of nodes.

    This list supports adding and removing nodes from the list during iteration.
    """

    # TODO(justinchuby): Make it a MutableSequence

    def __init__(self, root: TLinkable) -> None:
        # Using the root node simplifies the mutation implementation a lot
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

    def _insert_one_after(
        self,
        value: TLinkable,
        new_value: TLinkable,
        property_modifier: Callable[[TLinkable], None] | None = None,
    ) -> None:
        """Insert a new value after the given value.

        Example::
            Before: A -> B -> C
            Call: insert_after(B, D)
            After: A -> B -> D -> C
        Args:
            value: The value after which the new value is to be inserted.
            new_value: The new value to be inserted.
            property_modifier: A function that modifies the properties of the new node.
        """
        original_next = value._next
        value._next = new_value
        new_value._prev = new_value
        new_value._next = original_next
        original_next._prev = new_value
        # Un-erase the value in case it was previously erased
        new_value._erased = False
        self._length += 1
        if property_modifier is not None:
            property_modifier(value)

    def append(
        self, value: TLinkable, property_modifier: Callable[[TLinkable], None] | None = None
    ) -> None:
        """Append a node to the list."""
        self._insert_one_after(self._root._prev, value, property_modifier=property_modifier)

    def extend(
        self,
        values: Iterable[TLinkable],
        property_modifier: Callable[[TLinkable], None] | None = None,
    ) -> None:
        for value in values:
            self.append(value, property_modifier=property_modifier)

    def remove(
        self, value: TLinkable, property_modifier: Callable[[TLinkable], None] | None = None
    ) -> None:
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
        if property_modifier is not None:
            property_modifier(value)

    def insert_after(
        self,
        value: TLinkable,
        new_values: Iterable[TLinkable],
        property_modifier: Callable[[TLinkable], None] | None = None,
    ) -> None:
        """Insert new nodes after the given node.

        Args:
            value: The value after which the new values are to be inserted.
            new_values: The new values to be inserted.
            property_modifier: A function that modifies the properties of the new nodes.
        """
        insertion_point = value
        for new_value in new_values:
            self._insert_one_after(
                insertion_point, new_value, property_modifier=property_modifier
            )
            insertion_point = new_value

    def insert_before(
        self,
        value: TLinkable,
        new_values: Iterable[TLinkable],
        property_modifier: Callable[[TLinkable], None] | None = None,
    ) -> None:
        """Insert new nodes before the given node.

        Args:
            value: The value before which the new values are to be inserted.
            new_values: The new values to be inserted.
            property_modifier: A function that modifies the properties of the new nodes.
        """
        insertion_point = value._prev
        for new_value in new_values:
            self._insert_one_after(
                insertion_point, new_value, property_modifier=property_modifier
            )
            insertion_point = new_value
