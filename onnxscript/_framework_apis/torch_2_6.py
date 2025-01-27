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
    from onnxscript.onnx_opset._impl.opset18 import Opset18


def optimize(model: ir.Model) -> ir.Model:
    """Optimize the model."""
    optimizer.optimize_ir(model)
    return model


def torchlib_opset() -> Opset18:
    """Return the default opset for torchlib."""
    import onnxscript  # pylint: disable=import-outside-toplevel

    return onnxscript.opset18  # type: ignore


def torchlib_opset_version() -> int:
    """Return the default opset version for torchlib."""

    return torchlib_opset().version
