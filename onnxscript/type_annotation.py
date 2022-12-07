# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import typing

import onnx

from onnxscript.onnx_types import TensorType

_pytype_to_attrtype_map = {
    float: onnx.AttributeProto.FLOAT,
    int: onnx.AttributeProto.INT,
    str: onnx.AttributeProto.STRING,
}

_listtype_to_attrtype_map = {
    float: onnx.AttributeProto.FLOATS,
    int: onnx.AttributeProto.INTS,
    str: onnx.AttributeProto.STRINGS,
}

_list_constructors = [list, typing.List, typing.Sequence]


def _get_origin(t: type) -> typing.Optional[type]:
    """Substitute for typing.get_origin of Python 3.8+

    Note that the input t must be one of the valid types permitted for
    an input/attribute by ONNX Script.
    """
    if hasattr(typing, "get_origin"):
        return typing.get_origin(t)
    elif hasattr(t, "__origin__"):
        return t.__origin__  # type: ignore[no-any-return]
    else:
        return None


def _get_args(t: type) -> typing.Sequence[type]:
    """Substitute for typing.get_args of Python 3.8+"""
    if hasattr(typing, "get_args"):
        return typing.get_args(t)
    elif hasattr(t, "__args__"):
        return t.__args__  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported type annotation {t}")


def _get_element_type(t: type) -> type:
    """Returns the element type for a list or sequence type."""
    return _get_args(t)[0]


def pytype_to_attrtype(pytype: type) -> typing.Optional[onnx.AttributeProto.AttributeType]:
    if pytype in _pytype_to_attrtype_map:
        return _pytype_to_attrtype_map[pytype]
    if _get_origin(pytype) in _list_constructors:
        elt_type = _get_element_type(pytype)
        if elt_type in _listtype_to_attrtype_map:
            return _listtype_to_attrtype_map[elt_type]
    return None


def is_attr(pytype: type):
    return pytype_to_attrtype(pytype) is not None


def is_tensor(typeinfo):
    return isinstance(typeinfo, TensorType) or issubclass(typeinfo, TensorType)


def is_valid(typeinfo):
    return is_attr(typeinfo) or is_tensor(typeinfo)


def validate(typeinfo):
    if not is_valid(typeinfo):
        raise ValueError(f"Unsupported type annotation {typeinfo}")


def get_return_types(typeinfo: type | typing.Sequence[type]) -> typing.Sequence[type]:
    """Converts return-type annotation into a sequence of types.

    The return type annotation can be either a single type (for a single output)
    or a Tuple type (for multiple outputs). This function normalizes the
    representation so that it is always a sequence of types, even for a single
    output.
    """
    if isinstance(typeinfo, typing.Sequence):
        for t in typeinfo:
            validate(t)
        return typeinfo
    if _get_origin(typeinfo) == typing.Tuple:
        result = _get_args(typeinfo)
        for t in result:
            validate(t)
        return result
    validate(typeinfo)
    return (typeinfo,)
