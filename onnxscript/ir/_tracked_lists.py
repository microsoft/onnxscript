# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tracked lists for graph and node IO."""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from onnxscript.ir import _core

class NodeInputs(collections.UserList[_core.Value | None]):
    def __init__(self, node: _core.Node, initlist=None):
        super().__init__(initlist)
        self._node = node

    def append(self, value: _core.Value | None) -> None:
        """Add a new input to the node."""
        index = len(self.data)
        if value is not None:
            value._add_usage(self._node, index)  # pylint: disable=protected-access
        self.data.append(value)

    def insert(self, index: int, value: _core.Value | None) -> None:

    def unsupported(self, *_args, **_kwargs):
        raise RuntimeError("Method is not supported")

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
