"""Class for storing metadata about the IR objects."""

from __future__ import annotations

from typing import Any, Iterator, Mapping


class MetadataStore(Mapping[str, Any]):
    """Class for storing metadata about the IR objects.

    Metadata is stored as key-value pairs. The keys are strings and the values
    can be any Python object.

    The metadata store also supports marking keys as invalid. This is useful
    when a pass wants to mark a key that needs to be recomputed.
    """

    __slots__ = ["_store", "_invalid_keys"]

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._invalid_keys: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        # We do not check validity here because we want to allow
        # users to access invalid keys
        return self._store[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def items(self):
        return self._store.items()

    def keys(self):
        return self._store.keys()

    def values(self):
        return self._store.values()

    def __iter__(self) -> Iterator[str]:
        return self._store.__iter__()

    def __len__(self) -> int:
        return len(self._store)

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._invalid_keys.discard(key)

    def __contains__(self, key: str) -> bool:
        return key in self._store

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
        return f"{self.__class__.__name__}(store={self._store}, invalid_keys={self._invalid_keys})"

    def to_metadata_props(self) -> dict[str, str]:
        # TODO(justinchuby): Handle invalid keys
        return {k: str(v) for k, v in self.items()}
