# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stable APIs for PyTorch 2.6."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "optimize",
    "save_model_with_external_data",
    "torchlib_opset",
]
from typing import TYPE_CHECKING

from onnxscript import ir, optimizer
from onnxscript._framework_apis.torch_2_5 import (
    check_model,
    convert_version,
    get_torchlib_ops,
    save_model_with_external_data,
)

if TYPE_CHECKING:
    from onnxscript.values import Opset


def optimize(model: ir.Model) -> ir.Model:
    """Optimize the model."""
    optimizer.optimize_ir(model)
    return model


def torchlib_opset() -> Opset:
    """Return the default opset for torchlib."""
    from onnxscript import opset18

    return opset18
