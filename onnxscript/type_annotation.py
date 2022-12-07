# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import inspect
import typing

import onnx
from typing_extensions import get_args, get_origin

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


def pytype_to_attrtype(pytype: type) -> typing.Optional[onnx.AttributeProto.AttributeType]:
    if pytype in _pytype_to_attrtype_map:
        return _pytype_to_attrtype_map[pytype]
    if get_origin(pytype) in _list_constructors:
        args = get_args(pytype)
        elt_type = args[0]
        if elt_type in _listtype_to_attrtype_map:
            return _listtype_to_attrtype_map[elt_type]
    return None


def is_attr_type(pytype: type):
    return pytype_to_attrtype(pytype) is not None


def is_value_type(typeinfo):
    if isinstance(typeinfo, TensorType):
        return True
    if inspect.isclass(typeinfo) and issubclass(typeinfo, TensorType):
        return True
    return False


def is_valid_type(typeinfo):
    return is_attr_type(typeinfo) or is_value_type(typeinfo)


def get_return_types(typeinfo: type | typing.Sequence[type]) -> typing.Sequence[type]:
    """Converts return-type annotation into a sequence of types.

    The return type annotation can be either a single type (for a single output)
    or a Tuple type (for multiple outputs). This function normalizes the
    representation so that it is always a sequence of types, even for a single
    output.
    """
    if isinstance(typeinfo, typing.Sequence):
        return typeinfo
    if get_origin(typeinfo) is tuple:
        return get_args(typeinfo)
    return (typeinfo,)
