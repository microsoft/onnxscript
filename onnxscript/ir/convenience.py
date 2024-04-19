# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convenience methods for constructing (and manipulating?) the IR."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from onnxscript.ir import _core, _protocols, _enums


def convert_attribute(
    name: str,
    attr: str
    | int
    | float
    | Sequence[int]
    | Sequence[float]
    | Sequence[str]
    | _protocols.TensorProtocol
    | _core.Attr
    | None,
    attr_type: _enums.AttributeType | None= None,
) -> _core.Attr:
    if attr is None:
        if attr_type is None:
            raise ValueError("attr_type must be provided when attr is None")
        return _core.Attr(name, attr_type, None)
    if isinstance(attr, int):
        return _core.AttrInt64(name, attr)
    if isinstance(attr, float):
        return _core.AttrFloat32(name, attr)
    if isinstance(attr, str):
        return _core.AttrString(name, attr)
    if isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
        return _core.AttrInt64s(name, attr)  # type: ignore
    if isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
        return _core.AttrFloat32s(name, attr)  # type: ignore
    if isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
        return _core.AttrStrings(name, attr)  # type: ignore
    if isinstance(attr, (_core.Tensor, _protocols.TensorProtocol)):
        return _core.AttrTensor(name, attr)
    if isinstance(attr, _core.Attr):
        return attr
    raise TypeError(f"Unsupported attribute type: '{type(attr)}'")


def convert_attributes(attrs: Mapping[str, Any]) -> list[_core.Attr]:
    attributes: list[_core.Attr] = []
    for name, attr in attrs.items():
        attributes.append(convert_attribute(name, attr))
    return attributes
