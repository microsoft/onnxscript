# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Mutable list for nodes in a graph with safe mutation properties."""
# Disabled the following checks because this implementation makes heavy use of private members

# Inspired by https://github.com/pytorch/pytorch/blob/064a650b635e6fdaa8cf1a0dbc7dbbd23a37265d/torch/fx/graph.py

from __future__ import annotations

from typing import Generic, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


class _LinkBox(Generic[T]):
    """A link in a doubly linked list that has a reference to the actual object in the link.

    The :class:`_LinkBox` is a container for the actual object in the list. It is used to
    maintain the links between the elements in the linked list. The actual object is stored in the
    :attr:`value` attribute.

    By using a separate container for the actual object, we can safely remove the object from the
    list without losing the links. This allows us to remove the object from the list during
    iteration and place the object into a different list without breaking any chains.

    This is an internal class and should only be initialized by the :class:`DoublyLinkedHashList`.

    Attributes:
        prev: The previous box in the list.
        next: The next box in the list.
        erased: A flag to indicate if the box has been removed from the list.
        owning_list: The :class:`DoublyLinkedHashList` to which the box belongs.
        value: The actual object in the list.
    """

    __slots__ = ("prev", "next", "value", "owning_list")

    def __init__(self, owner: DoublyLinkedHashList[T], value: T | None) -> None:
        """Create a new link box.

        Args:
            owner: The linked list to which this box belongs.
            value: The value to be stored in the link box. When the value is None,
                the link box is considered erased (default). The root box of the list
                should be created with a None value.
        """
        self.prev: _LinkBox[T] = self
        self.next: _LinkBox[T] = self
        self.value: T | None = value
        self.owning_list: DoublyLinkedHashList[T] = owner

    @property
    def erased(self) -> bool:
        return self.value is None

    def erase(self) -> None:
        """Remove the link from the list and detach the value from the box."""
        if self.value is None:
            raise ValueError("_LinkBox is already erased")
        # Update the links
        prev, next_ = self.prev, self.next
        prev.next, next_.prev = next_, prev
        # Detach the value
        self.value = None

    def __repr__(self) -> str:
        return f"_LinkBox({self.value!r}, erased={self.erased}, prev={self.prev.value!r}, next={self.next.value!r})"


class DoublyLinkedHashList(Generic[T], Sequence[T]):
    """A doubly linked list of nodes.

    Adding and removing elements from the list during iteration is safe. Moving elements
    from one list to another is also safe.

    During the iteration:
    - If new elements are inserted after the current node, the iterator will
        iterate over them as well.
    - If new elements are inserted before the current node, they will
        not be iterated over in this iteration.
    - If the current node is lifted and inserted in a different location,
        iteration will start from the "next" node at the _original_ location.

    Time complexity:
        Inserting and removing nodes from the list is O(1). Accessing nodes by index is O(n),
        although accessing nodes at either end of the list is O(1). I.e. `list[0]` and `list[-1]`
        are O(1).
    """

    __slots__ = ("_root", "_length", "_values_to_boxes")

    def __init__(self, values: Iterable[T] | None = None) -> None:
        # Using the root node simplifies the mutation implementation a lot
        root_ = _LinkBox(self, None)
        self._root: _LinkBox = root_
        self._length = 0
        self._values_to_boxes: dict[T, _LinkBox] = {}
        if values is not None:
            self.extend(values)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements in the list.

        - If new elements are inserted after the current node, the iterator will
            iterate over them as well.
        - If new elements are inserted before the current node, they will
            not be iterated over in this iteration.
        - If the current node is lifted and inserted in a different location,
            iteration will start from the "next" node at the _original_ location.
        """
        box = self._root.next
        while box is not self._root:
            if box.owning_list is not self:
                raise RuntimeError(f"Element {box!r} is not in the list")
            if not box.erased:
                assert box.value is not None
                yield box.value
            box = box.next

    def __reversed__(self) -> Iterator[T]:
        """Iterate over the elements in the list in reverse order."""
        box = self._root.prev
        while box is not self._root:
            if not box.erased:
                assert box.value is not None
                yield box.value
            box = box.prev

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T:
        """Get the node at the given index.

        Complexity is O(n).
        """
        if index >= self._length or index < -self._length:
            raise IndexError(
                f"Index out of range: {index} not in range [-{self._length}, {self._length})"
            )
        if index < 0:
            # Look up from the end of the list
            iterator = reversed(self)
            item = next(iterator)
            for _ in range(-index - 1):
                item = next(iterator)
        else:
            iterator = iter(self)  # type: ignore[assignment]
            item = next(iterator)
            for _ in range(index):
                item = next(iterator)
        return item

    def _insert_one_after(
        self,
        box: _LinkBox[T],
        new_value: T,
    ) -> _LinkBox[T]:
        """Insert a new value after the given box.

        All insertion methods should call this method to ensure that the list is updated correctly.

        Example::
            Before: A  <->  B  <->  C
                    ^v0     ^v1     ^v2
            Call: _insert_one_after(B, v3)
            After:  A  <->  B  <->  new_box  <->  C
                    ^v0     ^v1       ^v3         ^v2

        Args:
            box: The box which the new value is to be inserted.
            new_value: The new value to be inserted.
            property_modifier: A function that modifies the properties of the new node.
        """
        if box.value is new_value:
            # Do nothing if the new value is the same as the old value
            return box
        if box.owning_list is not self:
            raise ValueError(f"Value {box.value!r} is not in the list")

        if new_value in self._values_to_boxes:
            # If the value is already in the list, remove it first
            self.remove(new_value)

        # Create a new _LinkBox for the new value
        new_box = _LinkBox(self, new_value)
        # original_box <=> original_next
        # becomes
        # original_box <=> new_box <=> original_next
        original_next = box.next
        box.next = new_box
        new_box.prev = box
        new_box.next = original_next
        original_next.prev = new_box

        # Be sure to update the length and mapping
        self._length += 1
        self._values_to_boxes[new_value] = new_box

        return new_box

    def _insert_many_after(
        self,
        box: _LinkBox[T],
        new_values: Iterable[T],
    ):
        """Insert multiple new values after the given box."""
        insertion_point = box
        for new_value in new_values:
            insertion_point = self._insert_one_after(insertion_point, new_value)

    def remove(self, value: T) -> None:
        """Remove a node from the list."""
        if value not in self._values_to_boxes:
            raise ValueError(f"Value {value!r} is not in the list")
        box = self._values_to_boxes[value]
        # Remove the link box and detach the value from the box
        box.erase()

        # Be sure to update the length and mapping
        self._length -= 1
        del self._values_to_boxes[value]

    def append(self, value: T) -> None:
        """Append a node to the list."""
        _ = self._insert_one_after(self._root.prev, value)

    def extend(
        self,
        values: Iterable[T],
    ) -> None:
        for value in values:
            self.append(value)

    def insert_after(
        self,
        value: T,
        new_values: Iterable[T],
    ) -> None:
        """Insert new nodes after the given node.

        Args:
            value: The value after which the new values are to be inserted.
            new_values: The new values to be inserted.
            property_modifier: A function that modifies the properties of the new nodes.
        """
        if value not in self._values_to_boxes:
            raise ValueError(f"Value {value!r} is not in the list")
        insertion_point = self._values_to_boxes[value]
        return self._insert_many_after(insertion_point, new_values)

    def insert_before(
        self,
        value: T,
        new_values: Iterable[T],
    ) -> None:
        """Insert new nodes before the given node.

        Args:
            value: The value before which the new values are to be inserted.
            new_values: The new values to be inserted.
            property_modifier: A function that modifies the properties of the new nodes.
        """
        if value not in self._values_to_boxes:
            raise ValueError(f"Value {value!r} is not in the list")
        insertion_point = self._values_to_boxes[value].prev
        return self._insert_many_after(insertion_point, new_values)
