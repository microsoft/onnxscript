# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stable APIs for PyTorch 2.8."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "optimize",
    "save_model_with_external_data",
]

import onnx_ir as ir

import onnxscript.optimizer
import onnxscript.rewriter.onnx_fusions
from onnxscript._framework_apis.torch_2_6 import (
    check_model,
    convert_version,
    get_torchlib_ops,
    save_model_with_external_data,
)


def optimize(model: ir.Model) -> ir.Model:
    """Optimize the model."""
    onnxscript.optimizer.optimize_ir(model)
    onnxscript.rewriter.onnx_fusions.fuse(model)
    return model
