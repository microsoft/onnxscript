"""Mutable list for nodes in a graph with safe mutation properties."""
# Disabled the following checks because this implementation makes heavy use of private members
# pylint: disable=protected-access

# Inspired by https://github.com/pytorch/pytorch/blob/064a650b635e6fdaa8cf1a0dbc7dbbd23a37265d/torch/fx/graph.py

from __future__ import annotations

from typing import Callable, Generic, Iterable, Iterator, Protocol, TypeVar


class Linkable(Protocol):
    # pylint: disable=unused-private-member
    _link_box: LinkBox | None
    # pylint: enable=unused-private-member


TLinkable = TypeVar("TLinkable", bound=Linkable)


class _EmptyValue(Linkable):
    def __init__(self, link_box: LinkBox | None = None) -> None:
        self._link_box: LinkBox | None = link_box


class LinkBox(Generic[TLinkable]):
    """A link in a doubly linked list that has a reference to the actual object in the link.

    A fields are private and are managed by DoublyLinkedList

    Attributes:
        prev: The previous element in the list.
        next: The next element in the list.
        erased: A flag to indicate if the element has been removed from the list.
        list: The DoublyLinkedList to which the element belongs.
    """

    def __init__(
        self, owner: DoublyLinkedList[TLinkable], value: TLinkable | _EmptyValue
    ) -> None:
        self.prev: LinkBox[TLinkable] = self
        self.next: LinkBox[TLinkable] = self
        value._link_box = self  # pylint: disable=protected-access  # type: ignore
        self.value: TLinkable | _EmptyValue = value
        self._list: DoublyLinkedList[TLinkable] = owner

    @property
    def erased(self) -> bool:
        return isinstance(self.value, _EmptyValue)

    def erase(self) -> None:
        """Remove the link from the list and detach the value from the box."""
        # Update the links
        prev, next_ = self.prev, self.next
        prev.next, next_.prev = next_, prev
        self.value._link_box = None  # pylint: disable=protected-access
        self.value = _EmptyValue(self)

    def __repr__(self) -> str:
        return f"LinkBox({self.value!r}, erased={self.erased}, prev={self.prev.value!r}, next={self.next.value!r})"


class DoublyLinkedList(Generic[TLinkable], Iterable[TLinkable]):
    """A doubly linked list of nodes.

    This list supports adding and removing nodes from the list during iteration.
    """

    def __init__(self) -> None:
        # Using the root node simplifies the mutation implementation a lot
        root_ = LinkBox(self, _EmptyValue())
        root_.value._link_box = root_  # pylint: disable=protected-access
        self._root: LinkBox = root_
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
        box = self._root.next
        while box is not self._root:
            if box._list is not self:
                raise RuntimeError(f"Element {box!r} is not in the list")
            if not box.erased:
                yield box.value
            box = box.next

    def __reversed__(self) -> Iterator[TLinkable]:
        """Iterate over the elements in the list in reverse order."""
        box = self._root.prev
        while box is not self._root:
            if not box.erased:
                yield box.value
            box = box.prev

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> TLinkable:
        """Get the node at the given index.

        Complexity is O(n).
        """
        if index >= len(self) or index < -len(self):
            raise IndexError("Index out of range")
        if index < 0:
            # Look up from the end of the list
            iterator = reversed(self)
            item = next(iterator)
            for _ in range(-index - 1):
                item = next(iterator)
            return item

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

        All insertion methods should call this method to ensure that the list is updated correctly.

        Example::
            Before: A -> B -> C
            Call: insert_after(B, D)
            After: A -> B -> D -> C
        Args:
            value: The value after which the new value is to be inserted.
            new_value: The new value to be inserted.
            property_modifier: A function that modifies the properties of the new node.
        """
        if value is new_value:
            # Do nothing if the new value is the same as the old value
            return
        if value._link_box is None or value._link_box._list is not self:
            raise ValueError(f"Value {value!r} is not in the list")
        assert value._link_box is not None
        # Remove the new value from the list if it is already in a different list
        if new_value._link_box is not None:
            new_value._link_box._list.remove(new_value)

        # Create a new LinkBox for the new value
        new_box = LinkBox(self, new_value)
        new_value._link_box = new_box
        # original_box <=> original_next
        # becomes
        # original_box <=> new_box <=> original_next
        original_box = value._link_box
        original_next = original_box.next
        original_box.next = new_box
        new_box.prev = original_box
        new_box.next = original_next
        original_next.prev = new_box

        # Call the property modifier in case the users want to modify the properties
        # For example, when a node is added to a graph, we want to set its graph property
        if property_modifier is not None:
            property_modifier(new_value)

        # Be sure to update the length
        self._length += 1

    def remove(
        self, value: TLinkable, property_modifier: Callable[[TLinkable], None] | None = None
    ) -> None:
        """Remove a node from the list."""
        if value._link_box is None:
            raise ValueError(f"Value {value!r} does not belong to any list")
        if value._link_box._list is not self:
            raise ValueError(f"Element {value!r} is not in the list")
        # Remove the link box and detach the value from the box
        value._link_box.erase()

        # Call the property modifier in case the users want to modify the properties
        # For example, when a node is removed from a graph, we want to unset its graph property
        if property_modifier is not None:
            property_modifier(value)

        # Be sure to update the length
        self._length -= 1

    def append(
        self, value: TLinkable, property_modifier: Callable[[TLinkable], None] | None = None
    ) -> None:
        """Append a node to the list."""
        self._insert_one_after(
            self._root.prev.value, value, property_modifier=property_modifier
        )  # type: ignore[arg-type]

    def extend(
        self,
        values: Iterable[TLinkable],
        property_modifier: Callable[[TLinkable], None] | None = None,
    ) -> None:
        for value in values:
            self.append(value, property_modifier=property_modifier)

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
        if value._link_box is None:
            raise ValueError(f"Value {value!r} does not belong to any list")
        insertion_point = value._link_box.prev.value
        for new_value in new_values:
            self._insert_one_after(
                insertion_point,  # type: ignore[arg-type]
                new_value,
                property_modifier=property_modifier,
            )
            insertion_point = new_value
