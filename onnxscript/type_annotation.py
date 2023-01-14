# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import collections
import inspect
import typing

import onnx
from typing_extensions import get_args, get_origin

from onnxscript.onnx_types import TensorType

_PYTYPE_TO_ATTRTYPE_MAP = {
    float: onnx.AttributeProto.FLOAT,
    int: onnx.AttributeProto.INT,
    str: onnx.AttributeProto.STRING,
    bool: onnx.AttributeProto.INT,
}

_LISTTYPE_TO_ATTRTYPE_MAP = {
    float: onnx.AttributeProto.FLOATS,
    int: onnx.AttributeProto.INTS,
    str: onnx.AttributeProto.STRINGS,
    bool: onnx.AttributeProto.INTS,
}

_LIST_CONSTRUCTORS = frozenset([list, typing.List, typing.Sequence, collections.abc.Sequence])


def pytype_to_attrtype(pytype: type) -> typing.Optional[onnx.AttributeProto.AttributeType]:
    if pytype in _PYTYPE_TO_ATTRTYPE_MAP:
        return _PYTYPE_TO_ATTRTYPE_MAP[pytype]
    if get_origin(pytype) in _LIST_CONSTRUCTORS:
        args = get_args(pytype)
        elt_type = args[0]
        if elt_type in _LISTTYPE_TO_ATTRTYPE_MAP:
            return _LISTTYPE_TO_ATTRTYPE_MAP[elt_type]
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
