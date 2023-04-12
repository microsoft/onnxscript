# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import collections
import inspect
import typing
from typing import Any, TypeVar, Union

import onnx
from typing_extensions import get_args, get_origin

from onnxscript import onnx_types

# TypeAnnotationValue represents the (value of) valid type-annotations recognized
# by ONNX Script. TODO: Flesh out a formal definition. Currently, it supports
# * float, int, str (primitive attribute types)
# * Sequence[float], Sequence[int], Sequence[str] (attribute types)
# * Tensor types
# * Sequence[Tensor] types
# * Union of above 2
# * TypeVars with above bounds
# * Above types with annotation attached
TypeAnnotationValue = typing.Any

# Map from python type to corresponding ONNX AttributeProto type
_PYTYPE_TO_ATTRTYPE_MAP = {
    float: onnx.AttributeProto.FLOAT,
    int: onnx.AttributeProto.INT,
    str: onnx.AttributeProto.STRING,
    bool: onnx.AttributeProto.INT,  # experimental
}

# Map from python type to corresponding ONNX AttributeProto type,
# for repeated (i.e., list of) values
_LISTTYPE_TO_ATTRTYPE_MAP = {
    float: onnx.AttributeProto.FLOATS,
    int: onnx.AttributeProto.INTS,
    str: onnx.AttributeProto.STRINGS,
    bool: onnx.AttributeProto.INTS,  # experimental
}

_LIST_CONSTRUCTORS = frozenset([list, typing.List, typing.Sequence, collections.abc.Sequence])


def _remove_annotation(typeinfo: TypeAnnotationValue) -> TypeAnnotationValue:
    """Remove Annotated wrapper if present, otherwise return typeinfo as is."""
    if hasattr(typing, "Annotated"):
        # Present in Python 3.9+
        if get_origin(typeinfo) is typing.Annotated:
            return get_args(typeinfo)[0]
    return typeinfo


def _is_primitive_attr_type(typeinfo: TypeAnnotationValue) -> bool:
    return typeinfo in _PYTYPE_TO_ATTRTYPE_MAP


def pytype_to_attrtype(
    pytype: TypeAnnotationValue,
) -> typing.Optional[onnx.AttributeProto.AttributeType]:
    pytype = _remove_annotation(pytype)
    if pytype in _PYTYPE_TO_ATTRTYPE_MAP:
        return _PYTYPE_TO_ATTRTYPE_MAP[pytype]
    type_constructor = get_origin(pytype)
    # Remove Optional wrapper if present, which is represented as an Union[..., type(None)]
    if type_constructor is typing.Union:
        # Filter out type(None), since typing.Optional[X] evaluates to Union[X, type(None)]
        args = [x for x in get_args(pytype) if x is not type(None)]
        if len(args) == 1:
            return pytype_to_attrtype(args[0])
    if type_constructor in _LIST_CONSTRUCTORS:
        elt_type = get_args(pytype)[0]
        if elt_type in _LISTTYPE_TO_ATTRTYPE_MAP:
            return _LISTTYPE_TO_ATTRTYPE_MAP[elt_type]
    return None


def _is_tensor_type(typeinfo: TypeAnnotationValue) -> bool:
    if isinstance(typeinfo, onnx_types.TensorType):
        return True
    if inspect.isclass(typeinfo) and issubclass(typeinfo, onnx_types.TensorType):
        return True
    return False


def is_value_type(typeinfo: TypeAnnotationValue) -> bool:
    """Returns True if typeinfo represents a value type, False if it is an attribute type.
    Raises ValueError if typeinfo is not a supported type annotation.
    """
    typeinfo = _remove_annotation(typeinfo)
    if _is_tensor_type(typeinfo):
        return True
    if _is_primitive_attr_type(typeinfo):
        return False
    type_constructor = get_origin(typeinfo)
    # Handle List-like type-constructor
    # Eg. List[INT32] is a value type, while List[int] is an attribute type
    if type_constructor in _LIST_CONSTRUCTORS:
        elt_type = get_args(typeinfo)[0]
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
            raise ValueError(f"Unsupported type annotation '{typeinfo}'")
    # Handle TypeVars:
    if isinstance(typeinfo, typing.TypeVar):
        if hasattr(typeinfo, "__bound__"):
            bound = typeinfo.__bound__
            return is_value_type(bound)
    raise ValueError(f"Unsupported type annotation {typeinfo}")


def is_attr_type(pytype: TypeAnnotationValue):
    return is_value_type(pytype) is False


def is_valid_type(typeinfo: TypeAnnotationValue):
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


def _reduce_type_var_to_union(hint: typing.TypeVar):
    """Reduce a TypeVar to a Union type on which we can use issubclass to check membership."""
    assert isinstance(hint, TypeVar)

    # If the TypeVar has a bound, use that.
    if hint.__bound__ is not None:
        return hint.__bound__

    # If the TypeVar has no bound, use the first constraint.
    if hint.__constraints__:
        return Union.__getitem__(hint.__constraints__)

    return Any


def get_supported_input_types(pytype) -> list[str]:
    """Returns a list of all supported input types for a given type annotation.

    Args:
        pytype: A type annotation.

    Returns:
        A list of all supported input types for the given type annotation.
    """
    supported_types: list[str] = []
    if typing.get_origin(pytype) is Union and isinstance(typing.get_args(pytype)[0], TypeVar):
        # Recursively unpack TypeVars inside an Optional
        for arg in typing.get_args(pytype):
            supported_types.extend(get_supported_input_types(arg))
        return supported_types

    if isinstance(pytype, TypeVar):
        pytype = _reduce_type_var_to_union(pytype)

    for tensor_type in onnx_types.ALL_TENSOR_TYPES:
        if isinstance(pytype, tensor_type):
            supported_types.append(tensor_type.to_string())
        if issubclass(tensor_type, pytype):
            supported_types.append(tensor_type.to_string())
        # TODO(justinchuby): Handle sequence types
        elif pytype == onnx_types.TensorType:
            supported_types.append(tensor_type.to_string())
        elif pytype is None:
            supported_types.append(tensor_type.to_string())
    return supported_types
