# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Optional, cast

import onnx


def make_attribute(
    key: str,
    value: Any,
    doc_string: Optional[str] = None,
    attr_type: Optional[int] = None,
) -> onnx.AttributeProto:
    """Helper function to create an ONNX AttributeProto.
    This is a temporary workaround until we can use an updated version of ONNX
    with PR 5220 fix, which adds attr_type as a parameter to helper.make_attribute.
    """

    if isinstance(value, list) and not value:
        # Temporary workaround for empty list:
        if attr_type is None:
            raise ValueError("Attribute type must be specified for empty list value.")
        proto = onnx.AttributeProto()
        proto.name = key
        proto.type = cast(onnx.AttributeProto.AttributeType, attr_type)
        return proto
    else:
        return onnx.helper.make_attribute(key, value, doc_string)
