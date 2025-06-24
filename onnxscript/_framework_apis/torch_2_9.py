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

import onnx_ir as ir

from onnxscript import optimizer
from onnxscript._framework_apis.torch_2_6 import (
    check_model,
    convert_version,
    get_torchlib_ops,
    save_model_with_external_data,
)
from onnxscript.rewriter import onnx_fusions


def optimize(model: ir.Model) -> ir.Model:
    """Optimize the model."""
    optimizer.optimize_ir(model)
    onnx_fusions.fuse(model)
    return model
