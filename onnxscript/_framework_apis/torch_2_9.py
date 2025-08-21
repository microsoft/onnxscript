# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stable APIs for PyTorch 2.9."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "optimize",
    "save_model_with_external_data",
]

from typing import TYPE_CHECKING

from onnxscript import version_converter
from onnxscript._framework_apis.torch_2_8 import (
    check_model,
    get_torchlib_ops,
    optimize,
    save_model_with_external_data,
)

if TYPE_CHECKING:
    import onnx_ir as ir


def convert_version(model: ir.Model, target_version: int) -> ir.Model:
    """Convert the model to the specified ONNX opset version.

    Starting from PyTorch 2.9, down conversion is turned on and supported.
    """
    version_converter.convert_version(model, target_version, fallback=True)
    return model
