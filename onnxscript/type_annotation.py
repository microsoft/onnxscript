# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import List, Optional, Sequence, get_args, get_origin

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

_list_constructors = {list, List, Sequence}


def pytype_to_attrtype(pytype: type) -> Optional[onnx.AttributeProto.AttributeType]:
    if pytype in _pytype_to_attrtype_map:
        return _pytype_to_attrtype_map[pytype]
    if get_origin(pytype) in _list_constructors:
        elt_type = get_args(pytype)[0]
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
