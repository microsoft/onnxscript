"""Convenience methods for constructing (and manipulating?) the IR."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from onnxscript.ir import _core


def convert_attributes(attrs: Mapping[str, Any]) -> list[_core.Attr]:
    attributes: list[_core.Attr] = []
    for name, attr in attrs.items():
        if isinstance(attr, int):
            attributes.append(_core.AttrInt64(name, attr))
        elif isinstance(attr, float):
            attributes.append(_core.AttrFloat32(name, attr))
        elif isinstance(attr, str):
            attributes.append(_core.AttrString(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
            attributes.append(_core.AttrInt64s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
            attributes.append(_core.AttrFloat32s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
            attributes.append(_core.AttrStrings(name, attr))
        elif isinstance(attr, _core.Attr):
            attributes.append(attr)
        else:
            raise TypeError(f"Unsupported attribute type: '{type(attr)}'")
    return attributes
