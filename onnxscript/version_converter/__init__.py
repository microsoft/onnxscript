# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    # Functions
    "convert_version",
]

from onnxscript import ir
from onnxscript.optimizer import _inliner
from onnxscript.version_converter import _version_converter


def convert_version(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""

    # In functions, we can have attribute-parameters, which means we don't know the value of the attribute.
    # Hence, we inline all the functions.
    _inliner.inline(model)
    _version_converter.convert_version(model, target_version)
