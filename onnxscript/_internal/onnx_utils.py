# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import onnx
from onnxscript._internal import autocast

_repeated_attribute_types = frozenset([
    onnx.AttributeProto.FLOATS,
    onnx.AttributeProto.INTS,
    onnx.AttributeProto.STRINGS,
    onnx.AttributeProto.TENSORS,
    onnx.AttributeProto.GRAPHS,
    onnx.AttributeProto.SPARSE_TENSORS,
    onnx.AttributeProto.TYPE_PROTOS,
])

def make_attribute(
    key: str,
    value: Any,
    name_generator: Callable[[], str],
    doc_string: Optional[str] = None,
    attr_type: Optional[int] = None,
) -> onnx.AttributeProto:
    """Helper function to create an ONNX AttributeProto.
    This is a refinement of onnx.helper.make_attribute that works with ONNX Script
    conventions for allowed types for attribute-values. In particular, it allows
    * Empty lists as attribute values, provided the attribute type is specified
    and is a list type.(ONNX PR 5220 fix provides the same extension to onnx.helper)
    * Allows scalar-values like 1.0 as well as lists like [1, -1] to be specified
    when the attribute type is TensorProto by automatically converting the value
    into a 0-D or 1-D tensor respectively.
    """

    if isinstance(value, list) and not value:
        # Empty list value:
        if attr_type is None:
            raise ValueError("Attribute type must be specified for empty list value.")
        attr_type_enum = cast(onnx.AttributeProto.AttributeType, attr_type)
        if attr_type_enum not in _repeated_attribute_types:
            raise ValueError(
                "Empty list value is only allowed for repeated attribute types."
            )
        proto = onnx.AttributeProto()
        proto.name = key
        proto.type = attr_type_enum
        return proto
    elif attr_type == onnx.AttributeProto.TENSOR:
        proto = onnx.AttributeProto()
        proto.name = key
        proto.type = attr_type_enum
        proto.t = autocast.pyvalue_to_onnx_tensor(name_generator(), value)
        return proto
    else:
        return onnx.helper.make_attribute(key, value, doc_string)
