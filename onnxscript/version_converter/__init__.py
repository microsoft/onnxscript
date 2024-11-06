# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    # Functions
    "convert_version"
]

from onnxscript import ir
from onnxscript.version_converter.version_converter import version_convert

BASE_OPSET_VERSION = 18


def convert_version(model: ir.Model, target_version: int, custom_adapters=None) -> ir.Model:
    """Convert the model to the specified ONNX opset version."""
    version_convert(model, target_version, custom_adapters=custom_adapters)
