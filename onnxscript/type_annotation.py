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


def pytype_to_attrtype(pytype) -> typing.Optional[onnx.AttributeProto.AttributeType]:
    if pytype in _PYTYPE_TO_ATTRTYPE_MAP:
        return _PYTYPE_TO_ATTRTYPE_MAP[pytype]
    # Remove Annotated wrapper if present
    if isinstance(pytype, typing._AnnotatedAlias):  # pylint: disable=protected-access
        return pytype_to_attrtype(get_args(pytype)[0])
    type_constructor = get_origin(pytype)
    # Remove Optional wrapper if present, which is represented as an Union[..., None]
    if type_constructor is typing.Union:
        # Filter out None, since typing.Optional[X] evaluates to Union[X, None]
        args = [x for x in get_args(pytype) if x is not type(None)]
        if len(args) == 1:
            return pytype_to_attrtype(args[0])
    if type_constructor in _LIST_CONSTRUCTORS:
        args = get_args(pytype)
        elt_type = args[0]
        if elt_type in _LISTTYPE_TO_ATTRTYPE_MAP:
            return _LISTTYPE_TO_ATTRTYPE_MAP[elt_type]
    return None


def is_tensor_type(typeinfo):
    if isinstance(typeinfo, TensorType):
        return True
    if inspect.isclass(typeinfo) and issubclass(typeinfo, TensorType):
        return True
    return False


def is_value_type(typeinfo) -> bool:
    """Returns True if typeinfo represents a value type, False if it is an attribute type.
    Raises ValueError if typeinfo is not a supported type annotation.
    """
    # Remove Annotated wrapper if present
    if isinstance(typeinfo, typing._AnnotatedAlias):  # pylint: disable=protected-access
        typeinfo = get_args(typeinfo)[0]
    if is_tensor_type(typeinfo):
        return True
    if is_primitive_attr_type(typeinfo):
        return False
    type_constructor = get_origin(typeinfo)
    # Handle List-like type-constructor
    # Eg. List[INT32] is a value type, while List[int] is an attribute type
    if type_constructor in _LIST_CONSTRUCTORS:
        args = get_args(typeinfo)
        elt_type = args[0]
        return is_value_type(elt_type)
    # Handle Union and Optional type-constructors
    if type_constructor is typing.Union:
        # Filter out None, since typing.Optional[X] evaluates to Union[X, None]
        args = [x for x in get_args(typeinfo) if x is not type(None)]
        args_value_check = [is_value_type(x) for x in args]
        if all(args_value_check):
            # Handles cases like Optional[INT32] as well as Union[FLOAT16, FLOAT, DOUBLE]
            return True
        elif (len(args) == 1) and args_value_check[0] is False:
            # Handle the case of optional attribute: eg. Optional[int]
            # Note that we do not allow Union[int, float] for attributes.
            return False
        else:
            raise ValueError(f"Unsupported type annotation {typeinfo}")
    # Handle TypeVars:
    if isinstance(typeinfo, typing.TypeVar):
        if hasattr(typeinfo, "__bound__"):
            bound = typeinfo.__bound__
            return is_value_type(bound)
    raise ValueError(f"Unsupported type annotation {typeinfo}")


def is_attr_type(pytype: type):
    return is_value_type(pytype) is False


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
