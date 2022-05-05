# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import typing
from typing import List

pytype_to_attrtype_map = {
    float: onnx.AttributeProto.FLOAT,
    int: onnx.AttributeProto.INT,
    str: onnx.AttributeProto.STRING,
    typing.List[int]: onnx.AttributeProto.INTS
}


def is_attr(typeinfo):
    return typeinfo in {float, int, str, List[int]}
    # (typeinfo is float) or (typeinfo is str) or (typeinfo is int)


def is_tensor(typeinfo):
    return hasattr(typeinfo, "to_type_proto")
    # return isinstance(typeinfo, onnxscript.Tensor)  # TODO


def is_valid(typeinfo):
    return is_attr(typeinfo) or is_tensor(typeinfo)
