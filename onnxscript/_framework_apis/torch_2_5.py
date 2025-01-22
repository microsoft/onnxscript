# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stable APIs for PyTorch 2.5."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "optimize",
    "save_model_with_external_data",
]

import dataclasses
import os
import pathlib
from typing import Callable

from onnxscript import ir, optimizer, version_converter
from onnxscript.function_libs.torch_lib import registration


@dataclasses.dataclass(frozen=True)
class _OnnxFunctionMeta:
    """A wrapper of onnx-script function with additional metadata.

    qualified_name: The qualified name of the aten operator.
    function: The onnx-script function.
    domain: The domain of the function.
    name: The name of the function.
    is_complex: Whether the function is a complex function.
    """

    qualified_name: str
    function: Callable
    domain: str
    name: str
    is_complex: bool = False


def optimize(model: ir.Model) -> ir.Model:
    """Optimize the model."""
    # Internal flag. Will go away.
    enabled = os.getenv("TORCH_ONNX_ENABLE_OPTIMIZATION") == "1"
    if enabled:
        optimizer.optimize_ir(model)
    return model


def convert_version(model: ir.Model, target_version: int) -> ir.Model:
    """Convert the model to the specified ONNX opset version."""
    # Internal flag. Will go away.
    enabled = os.getenv("TORCH_ONNX_ENABLE_VERSION_CONVERSION") == "1"
    if enabled:
        version_converter.convert_version(model, target_version)
    return model


def check_model(model: ir.Model) -> None:
    """Check the model."""

    del model  # Unused yet


def save_model_with_external_data(model: ir.Model, model_path: str | os.PathLike) -> None:
    """Save the model with external data. The model is unchanged after saving."""

    # TODO(#1835): Decide if we want to externalize large attributes as well
    for value in model.graph.initializers.values():
        if value.const_value is None:
            raise ValueError(
                "The model contains uninitialized initializer values. "
                "Please make sure all initializer values are initialized."
            )
    destination_path = pathlib.Path(model_path)
    data_path = f"{destination_path.name}.data"

    ir.save(model, model_path, external_data=data_path)


def get_torchlib_ops() -> list[_OnnxFunctionMeta]:
    # Trigger op registration
    from onnxscript.function_libs.torch_lib import (  # pylint: disable=import-outside-toplevel
        ops,
    )

    del ops  # Unused

    torchlib_registry = registration.default_registry
    function_metas = []

    for qualified_name, aten_overloads_func in torchlib_registry.items():
        if qualified_name.startswith("internal::"):
            # Skip the custom defined internal functions
            continue

        for overload_func in aten_overloads_func.overloads:
            function_meta = _OnnxFunctionMeta(
                qualified_name=qualified_name,
                function=overload_func,
                domain=overload_func.function_ir.domain,
                name=overload_func.name,
                is_complex=False,
            )
            function_metas.append(function_meta)
        for complex_func in aten_overloads_func.complex:
            function_meta = _OnnxFunctionMeta(
                qualified_name=qualified_name,
                function=complex_func,
                domain=complex_func.function_ir.domain,
                name=complex_func.name,
                is_complex=True,
            )
            function_metas.append(function_meta)

    return function_metas
