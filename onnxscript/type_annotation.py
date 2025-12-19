# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Public API for type_annotation module.

This module re-exports the public API from the internal type_annotation module.
"""

from onnxscript._internal.type_annotation import (
    ALL_TENSOR_TYPE_STRINGS,
    TypeAnnotationValue,
    get_type_constraint_name,
    is_attr_type,
    is_optional,
    is_valid_type,
    is_value_type,
    onnx_attr_type_to_onnxscript_repr,
    pytype_to_attrtype,
    pytype_to_type_strings,
)

__all__ = [
    "ALL_TENSOR_TYPE_STRINGS",
    "TypeAnnotationValue",
    "get_type_constraint_name",
    "is_attr_type",
    "is_optional",
    "is_valid_type",
    "is_value_type",
    "onnx_attr_type_to_onnxscript_repr",
    "pytype_to_attrtype",
    "pytype_to_type_strings",
]
