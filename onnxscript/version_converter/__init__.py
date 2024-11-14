# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    # Functions
    "convert_version",
    "inline",
]

from onnxscript import ir
from onnxscript.optimizer._inliner import inline
from onnxscript.version_converter.version_converter import version_convert


def convert_version(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""

    # In functions, we can have attribute-parameters, which means we don't know the value of the attribute.
    # Hence, we inline all the functions.
    inline(model)
    version_convert(model, target_version)
