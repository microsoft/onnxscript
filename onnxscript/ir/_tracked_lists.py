# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tracked lists for graph and node IO."""

from __future__ import annotations

import collections
import contextlib
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from onnxscript.ir import _core


@contextlib.contextmanager
def _update_value_usages(node: _core.Node, inputs: list[_core.Value | None]):
    """Temporarily unset usages of the inputs."""
    for i, item in enumerate(inputs):
        if item is not None:
            item._remove_usage(node, i)  # pylint: disable=protected-access
    try:
        # Caller will modify the inputs
        yield
    finally:
        for i, item in enumerate(inputs):
            if item is not None:
                item._add_usage(node, i)  # pylint: disable=protected-access


class NodeInputs(collections.UserList[_core.Value | None]):
    def __init__(self, node: _core.Node, initlist=None):
        super().__init__(initlist)
        self._node = node

    def prepend(self, item: _core.Value | None) -> None:
        """Add a new input to the node."""
        self.insert(0, item)

    def append(self, item: _core.Value | None) -> None:
        """Add a new input to the node."""
        index = len(self.data)
        if item is not None:
            item._add_usage(self._node, index)  # pylint: disable=protected-access
        self.data.append(item)

    def extend(self, other) -> None:
        for item in other:
            self.append(item)

    def insert(self, i: int, item: _core.Value | None) -> None:
        with _update_value_usages(self._node, self.data):
            self.data.insert(i, item)

    def pop(self, i: int = -1) -> _core.Value | None:
        item = self.data[i]
        if i == -1:
            # Remove the last item. No usages need to be updated
            if item is not None:
                item._remove_usage(self._node, i)
            return self.data.pop()
        # Otherwise we need to update usages
        with _update_value_usages(self._node, self.data):
            result = self.data.pop(i)

        return result

    def clear(self) -> None:
        """Clear the list."""
        for i, item in enumerate(self.data):
            if item is not None:
                item._remove_usage(self._node, i)
        self.data.clear()

    def __setitem__(self, i: int, item: _core.Value | None) -> None:
        """Replace an input to the node."""
        if i < -len(self.data) or i >= len(self.data):
            raise ValueError(f"index out of range: {i}")
        if i < 0:
            i += len(self.data)
        assert i >= 0
        old_input = self.data[i]
        if old_input is not None:
            old_input._remove_usage(self._node, i)  # pylint: disable=protected-access
        if item is not None:
            item._add_usage(self._node, i)  # pylint: disable=protected-access
        self.data[i] = item

    def unsupported(self, *_args, **_kwargs):
        raise NotImplementedError("Method is not supported")

    __lt__ = unsupported
    __le__ = unsupported
    __gt__ = unsupported
    __ge__ = unsupported
    __add__ = unsupported
    __radd__ = unsupported
    __iadd__ = unsupported
    __mul__ = unsupported
    reverse = unsupported
    sort = unsupported


class NodeOutputs(collections.UserList[_core.Value]):
    def __init__(self, node: _core.Node, initlist=None):
        super().__init__(initlist)
        self._node = node

    def append(self, item: _core.Value) -> None:
        """Add a new output to the node."""
        if item.producer() is not None and item.producer() is not self._node:
            raise NotImplementedError(
                f"Output already has a producer that is not this node ({item.producer()}). "
                "An output value can be owned by only one node throughout its lifetime. "
                "Instead, create a new value and assign it to the output. Replace all usages of the old value with the new one."
            )
        item._producer = self._node  # pylint: disable=protected-access
        item._index = len(self.data)  # pylint: disable=protected-access
        self.data.append(item)

    def extend(self, other) -> None:
        for item in other:
            self.append(item)

    def pop(self, i: int = -1) -> _core.Value | None:
        item = self.data[i]
        if i == -1:
            # Remove the last item. No usages need to be updated
            if item.uses():
                raise ValueError(
                    f"Cannot remove output {item} because it is still used by other nodes."
                )
            return self.data.pop()

        for j, output in enumerate(self.data):
            output._index = j  # pylint: disable=protected-access

    def clear(self) -> None:
        """Clear the list."""
        for _ in range(len(self.data)):
            self.pop()

    def __setitem__(self, i: int, item: _core.Value) -> None:
        if item is None:
            raise NotImplementedError(
                "An output cannot be None. To remove a trailing output, use pop(). "
                "To remove an output in the middle, set its name to an empty string instead."
            )
        self.pop(i)
        self.insert(i, item)

    def insert(self, i: int, item: _core.Value) -> None:
        """Replace an output to the node."""
        if i < -len(self.data) or i >= len(self.data):
            raise ValueError(f"index out of range: {i}")
        if i < 0:
            i += len(self.data)
        assert i >= 0
        if item.producer() is not None and item.producer() is not self._node:
            raise NotImplementedError(
                f"Output already has a producer that is not this node ({item.producer()}). "
                "An output value can be owned by only one node throughout its lifetime. "
                "Instead, create a new value and assign it to the output. Replace all usages of the old value with the new one."
            )

        item._producer = self._node  # pylint: disable=protected-access

        # Update the index of the item being replaced
        self.data.insert(i, item)
        for j, output in enumerate(self.data):
            output._index = j  # pylint: disable=protected-access


    def unsupported(self, *_args, **_kwargs):
        raise NotImplementedError("Method is not supported")

    __lt__ = unsupported
    __le__ = unsupported
    __gt__ = unsupported
    __ge__ = unsupported
    __add__ = unsupported
    __radd__ = unsupported
    __iadd__ = unsupported
    __mul__ = unsupported
    # NOTE: We don't support insertion and removal of items in the middle of the list
    # because in ONNX outputs are positional and index dependent. To remove an output in
    # the middle, set its name to an empty empty string instead.
    reverse = unsupported
    sort = unsupported
    remove = unsupported
