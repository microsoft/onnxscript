# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stable APIs for PyTorch 2.7."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "optimize",
    "save_model_with_external_data",
]

from onnxscript._framework_apis.torch_2_6 import (
    check_model,
    convert_version,
    get_torchlib_ops,
    optimize,
    save_model_with_external_data,
)
