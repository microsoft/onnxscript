# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    # Functions
    "convert_version",
]

from onnxscript import ir
from onnxscript.ir.passes.common import version_converter as _version_converter_pass


def convert_version(model: ir.Model, target_version: int, fallback=False) -> None:
    """Convert the model to the specified ONNX opset version.

    Args:
        model: The model to convert.
        target_version: The target ONNX opset version.
        fallback: Whether to fallback to the onnx version converter if the
            target version is not supported. Default is True.
    """
    # In functions, we can have attribute-parameters, which means we don't know the value of the attribute.
    # Hence, we inline all the functions.
    _version_converter_pass.ConvertVersionPass(target_version=target_version, fallback=fallback)(model)
