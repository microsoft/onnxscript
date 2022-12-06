# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import typing

import onnx

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
        return t.__origin__
    else:
        return None


def _get_element_type(t: type) -> type:
    """Returns the element type for a list or sequence type."""
    if hasattr(typing, "get_args"):
        return typing.get_args(t)[0]
    elif hasattr(t, "__args__"):
        return t.__args__[0]
    else:
        raise ValueError(f"Cannot get element type from {t}")


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
    return hasattr(typeinfo, "to_type_proto")
    # return isinstance(typeinfo, onnxscript.Tensor)  # TODO


def is_valid(typeinfo):
    return is_attr(typeinfo) or is_tensor(typeinfo)
