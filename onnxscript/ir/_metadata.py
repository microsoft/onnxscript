# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Class for storing metadata about the IR objects."""

from __future__ import annotations

import collections
from typing import Any, Mapping


class MetadataStore(collections.UserDict):
    """Class for storing metadata about the IR objects.

    Metadata is stored as key-value pairs. The keys are strings and the values
    can be any Python object.

    The metadata store also supports marking keys as invalid. This is useful
    when a pass wants to mark a key that needs to be recomputed.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, /) -> None:
        super().__init__(data)
        self._invalid_keys: set[str] = set()

    def __setitem__(self, key: str, item: Any) -> None:
        self.data[key] = item
        self._invalid_keys.discard(key)

    def invalidate(self, key: str) -> None:
        self._invalid_keys.add(key)

    def is_valid(self, key: str) -> bool:
        """Returns whether the value is valid.

        Note that default values (None) are not necessarily invalid. For example,
        a shape that is unknown (None) may be still valid if shape inference has
        determined that the shape is unknown.

        Whether a value is valid is solely determined by the user that sets the value.
        """
        return key not in self._invalid_keys

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data!r}, invalid_keys={self._invalid_keys!r})"
