"""Stable APIs for PyTorch 2.5."""

from __future__ import annotations

__all__ = [
    "check_model",
    "convert_version",
    "get_torchlib_ops",
    "OnnxFunctionMeta",
    "optimize",
    "save_model_with_external_data",
]

import dataclasses
import os
import pathlib
from typing import Callable

import onnx
from onnxscript import ir


@dataclasses.dataclass(frozen=True)
class OnnxFunctionMeta:
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

    # TODO(justinchuby): Use the optimizer
    return model


def convert_version(model: ir.Model, target_version: int) -> ir.Model:
    """Convert the model to the specified ONNX opset version."""
    proto = ir.serde.serialize_model(model)
    proto = onnx.version_converter.convert_version(proto, target_version)
    return ir.serde.deserialize_model(proto)


def check_model(model: ir.Model) -> None:
    """Check the model."""
    pass


def save_model_with_external_data(
    model: ir.Model, model_path: str | os.PathLike
) -> None:
    """Save the model with external data."""

    destination_path = pathlib.Path(model_path)
    # Create the directory if it does not exist
    data_path = f"{destination_path.name}.data"
    proto = ir.serde.serialize_model(model)
    onnx.save_model(
        proto,
        model_path,
        save_as_external_data=True,
        location=data_path,
    )


def get_torchlib_ops() -> list[OnnxFunctionMeta]:
    from onnxscript.function_libs.torch_lib import (
        registration as torchlib_registration,
    )

    torchlib_registry = torchlib_registration.default_registry
    function_metas = []

    for qualified_name, aten_overloads_func in torchlib_registry.items():
        if qualified_name.startswith("internal::"):
            # Skip the custom defined internal functions
            continue

        for overload_func in aten_overloads_func.overloads:
            function_meta = OnnxFunctionMeta(
                qualified_name=qualified_name,
                function=overload_func,
                domain=overload_func.function_ir.domain,
                name=overload_func.name,
                is_complex=False,
            )
            function_metas.append(function_meta)
        for complex_func in aten_overloads_func.complex:
            qualified_name = complex_func.name
            function_meta = OnnxFunctionMeta(
                qualified_name=qualified_name,
                function=complex_func,
                domain=complex_func.function_ir.domain,
                name=complex_func.name,
                is_complex=True,
            )
            function_metas.append(function_meta)

    return function_metas
