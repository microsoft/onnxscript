# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing and manipulating the IR."""

from __future__ import annotations

__all__ = [
    "convert_attribute",
    "convert_attributes",
    "replace_all_uses_with",
]

from onnxscript.ir._convenience import (
    convert_attribute,
    convert_attributes,
    replace_all_uses_with,
)

# NOTE: Do not implement any other functions in this module.
# implement them in the _convenience module and import them here instead.


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
