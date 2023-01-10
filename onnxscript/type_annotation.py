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
}

_LISTTYPE_TO_ATTRTYPE_MAP = {
    float: onnx.AttributeProto.FLOATS,
    int: onnx.AttributeProto.INTS,
    str: onnx.AttributeProto.STRINGS,
}

_LIST_CONSTRUCTORS = frozenset([list, typing.List, typing.Sequence, collections.abc.Sequence])


def is_primitive_attr_type(typeinfo) -> bool:
    return typeinfo in _PYTYPE_TO_ATTRTYPE_MAP

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


def is_tensor_type(typeinfo):
    if isinstance(typeinfo, TensorType):
        return True
    if inspect.isclass(typeinfo) and issubclass(typeinfo, TensorType):
        return True
    return False

def is_value_type(typeinfo):
    if is_tensor_type(typeinfo):
        return True
    if is_primitive_attr_type(typeinfo):
        return False
    type_constructor = get_origin(typeinfo)
    if type_constructor in _LIST_CONSTRUCTORS:
        args = get_args(typeinfo)
        elt_type = args[0]
        return is_value_type(elt_type)
    if type_constructor is typing.Optional:
        args = get_args(typeinfo)
        elt_type = args[0]
        return is_value_type(elt_type)
    raise ValueError(f"Unsupported type annotation {typeinfo}")        


def is_valid_type(typeinfo):
    try:
        return is_value_type(typeinfo) in {True, False}
    except ValueError:
        return False

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
