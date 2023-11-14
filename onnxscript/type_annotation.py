# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import collections
import inspect
import typing
from typing import Optional, Sequence, Union

import onnx

from onnxscript import onnx_types

# TypeAnnotationValue represents the (value of) valid type-annotations recognized
# by ONNX Script. TODO: Flesh out a formal definition. Currently, it supports
# - float, int, str (primitive attribute types)
# - Sequence[float], Sequence[int], Sequence[str] (attribute types)
# - Tensor types
# - Sequence[Tensor] types
# - Union of above 2
# - TypeVars with above bounds
# - Above types with annotation attached
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

# Map from ONNX AttributeProto type to its representation (in ONNX Script).
_ATTRTYPE_TO_REPR = {
    onnx.AttributeProto.FLOAT: "float",
    onnx.AttributeProto.INT: "int",
    onnx.AttributeProto.STRING: "str",
    onnx.AttributeProto.FLOATS: "Sequence[float]",
    onnx.AttributeProto.INTS: "Sequence[int]",
    onnx.AttributeProto.STRINGS: "Sequence[str]",
}


def onnx_attr_type_to_onnxscript_repr(attr_type: onnx.AttributeProto.AttributeType) -> str:
    if attr_type not in _ATTRTYPE_TO_REPR:
        supported = ", ".join(
            f"'{onnx.AttributeProto.AttributeType.Name(v)}'" for v in _ATTRTYPE_TO_REPR
        )
        raise ValueError(f"Unsupported attribute type {attr_type}: only {supported} allowed.")
    return _ATTRTYPE_TO_REPR[attr_type]


# A sorted list of all type strings used in an OpSchema
ALL_TENSOR_TYPE_STRINGS = tuple(
    sorted(tensor_type.to_string() for tensor_type in onnx_types.tensor_type_registry.values())
)


def _remove_annotation(typeinfo: TypeAnnotationValue) -> TypeAnnotationValue:
    """Remove Annotated wrapper if present, otherwise return typeinfo as is."""
    if hasattr(typing, "Annotated"):
        # Present in Python 3.9+
        if typing.get_origin(typeinfo) is typing.Annotated:
            return typing.get_args(typeinfo)[0]
    return typeinfo


def _is_primitive_attr_type(typeinfo: TypeAnnotationValue) -> bool:
    return typeinfo in _PYTYPE_TO_ATTRTYPE_MAP


def pytype_to_attrtype(
    pytype: TypeAnnotationValue,
) -> Optional[onnx.AttributeProto.AttributeType]:
    pytype = _remove_annotation(pytype)
    if pytype in _PYTYPE_TO_ATTRTYPE_MAP:
        return _PYTYPE_TO_ATTRTYPE_MAP[pytype]
    type_constructor = typing.get_origin(pytype)
    # Remove Optional wrapper if present, which is represented as an Union[..., type(None)]
    if type_constructor is typing.Union:
        # Filter out type(None), since typing.Optional[X] evaluates to Union[X, type(None)]
        args = [x for x in typing.get_args(pytype) if x is not type(None)]
        if len(args) == 1:
            return pytype_to_attrtype(args[0])
    if type_constructor in _LIST_CONSTRUCTORS:
        elt_type = typing.get_args(pytype)[0]
        if elt_type in _LISTTYPE_TO_ATTRTYPE_MAP:
            return _LISTTYPE_TO_ATTRTYPE_MAP[elt_type]
    return None


def base_type_is_bool(pytype: TypeAnnotationValue) -> bool:
    """Returns True if base type of pytype is bool, False otherwise."""
    pytype = _remove_annotation(pytype)
    if pytype in _PYTYPE_TO_ATTRTYPE_MAP:
        return pytype is bool
    type_constructor = typing.get_origin(pytype)
    if type_constructor in _LIST_CONSTRUCTORS:
        element_type = typing.get_args(pytype)[0]
        return element_type is bool
    # Remove Optional wrapper if present:
    if type_constructor is Optional or type_constructor is Union:
        # In Python < 3.10, Optional[X] is represented as Union[X, type(None)]
        # so we filter out type(None) if present
        args = [x for x in typing.get_args(pytype) if x is not type(None)]
        if len(args) == 1:
            return base_type_is_bool(args[0])

    return False


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
    type_constructor = typing.get_origin(typeinfo)
    # Handle List-like type-constructor
    # Eg. List[INT32] is a value type, while List[int] is an attribute type
    if type_constructor in _LIST_CONSTRUCTORS:
        elt_type = typing.get_args(typeinfo)[0]
        return is_value_type(elt_type)
    # Handle Union and Optional type-constructors
    if type_constructor is typing.Union:
        # Filter out None, since typing.Optional[X] evaluates to Union[X, None]
        args = [x for x in typing.get_args(typeinfo) if x is not type(None)]
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


def is_optional(pytype) -> bool:
    """Returns whether a pytype is an Optional."""
    if typing.get_origin(pytype) is Union and type(None) in typing.get_args(pytype):
        # Python < 3.10
        return True
    if typing.get_origin(pytype) is Optional:
        # Python >= 3.10
        return True
    return False


def get_return_types(typeinfo: type | Sequence[type]) -> Sequence[type]:
    """Converts return-type annotation into a sequence of types.

    The return type annotation can be either a single type (for a single output)
    or a Tuple type (for multiple outputs). This function normalizes the
    representation so that it is always a sequence of types, even for a single
    output.
    """
    if isinstance(typeinfo, typing.Sequence):
        return typeinfo
    if typing.get_origin(typeinfo) is tuple:
        return typing.get_args(typeinfo)
    return (typeinfo,)


def pytype_to_type_strings(pytype: TypeAnnotationValue) -> list[str]:
    """Returns a list of type-strings corresponding to a given type annotation.

    Args:
        pytype: A type annotation.

    Returns:
        A list of all supported input types for the given type annotation.
        Ensures that the list is sorted in the same order as ALL_TYPE_STRINGS.
    """
    if pytype is None:
        return list(ALL_TENSOR_TYPE_STRINGS)
    if pytype is onnx_types.TensorType:
        return list(ALL_TENSOR_TYPE_STRINGS)
    if isinstance(pytype, type) and issubclass(pytype, onnx_types.TensorType):
        return [pytype.to_string()]
    if isinstance(pytype, onnx_types.TensorType):
        return [pytype.to_string()]
    if isinstance(pytype, typing.TypeVar):
        constraints = pytype.__constraints__
        if constraints:
            return pytype_to_type_strings(Union.__getitem__(constraints))  # pylint: disable=unnecessary-dunder-call
        bound = pytype.__bound__
        if bound is None:
            return list(ALL_TENSOR_TYPE_STRINGS)
        return pytype_to_type_strings(bound)
    if typing.get_origin(pytype) is Union:
        options = []
        subtypes = typing.get_args(pytype)
        # A None type in a Union is equivalent to an optional type
        optional = is_optional(pytype)
        for subtype in subtypes:
            if subtype is type(None):
                # Skip None type because we are handling it with is_optional
                continue
            if optional:
                options += [
                    *pytype_to_type_strings(subtype),
                    *[f"optional({s})" for s in pytype_to_type_strings(subtype)],
                ]
            else:
                options += pytype_to_type_strings(subtype)
        # Remove duplicates
        return sorted(set(options))
    if typing.get_origin(pytype) in _LIST_CONSTRUCTORS:
        subtypes = typing.get_args(pytype)
        return [f"seq({s})" for s in pytype_to_type_strings(subtypes[0])]

    raise ValueError(f"Unsupported type: {pytype}")


def get_type_constraint_name(pytype: TypeAnnotationValue) -> Optional[str]:
    """Returns the name of the type constraint for a given type annotation.

    Args:
        pytype: A type annotation.

    Returns:
        The name of the type constraint if it is a TypeVar.
        - Prefixes the name with "Optional_" if the type annotation is Optional[TypeVar].
        - Prefixes the name with "Sequence_" if the type annotation is a Sequence[].
        - Returns None if the type annotation does not have a type constraint.
    """
    if isinstance(pytype, typing.TypeVar):
        return pytype.__name__
    if is_optional(pytype):
        subtypes = typing.get_args(pytype)
        for subtype in subtypes:
            if subtype is type(None):
                continue
            type_param_name = get_type_constraint_name(subtype)
            return f"Optional_{type_param_name}" if type_param_name else None
    if typing.get_origin(pytype) in _LIST_CONSTRUCTORS:
        subtypes = typing.get_args(pytype)
        type_param_name = get_type_constraint_name(subtypes[0])
        return f"Sequence_{type_param_name}" if type_param_name else None
    return None
