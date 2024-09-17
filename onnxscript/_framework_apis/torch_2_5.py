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

import onnx

from onnxscript import ir, optimizer
from onnxscript.function_libs.torch_lib import registration
from onnxscript.ir import _external_data

# Internal flag. Will go away.
_TORCH_ONNX_SAVE_EXTERNAL_DATA_WITH_IR = (
    os.getenv("TORCH_ONNX_OFFLOAD_EXTERNAL_DATA_WITH_IR") != "0"
)


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
    # model_version = model.opset_import.get("")
    # if model_version == target_version:
    #     # No conversion needed
    #     return model

    # # FIXME(justinchuby): version_converter does not support functions
    # proto = ir.serde.serialize_model(model)
    # proto = onnx.version_converter.convert_version(proto, target_version)
    # return ir.serde.deserialize_model(proto)
    # TODO(justinchuby): This function needs to be carefully implemented
    # to handle large models. For now, we just return the model.
    del target_version  # Unused
    return model


def check_model(model: ir.Model) -> None:
    """Check the model."""

    del model  # Unused yet


def save_model_with_external_data(model: ir.Model, model_path: str | os.PathLike) -> None:
    """Save the model with external data. The model is unchanged after saving."""

    # TODO(#1835): Decide if we want to externalize large attributes as well
    if _TORCH_ONNX_SAVE_EXTERNAL_DATA_WITH_IR:
        initializer_values = tuple(model.graph.initializers.values())
        tensors = [v.const_value for v in initializer_values]
        for tensor in tensors:
            if tensor is None:
                raise ValueError(
                    "The model contains uninitialized initializer values. "
                    "Please make sure all initializer values are initialized."
                )
        destination_path = pathlib.Path(model_path)
        base_dir = destination_path.parent
        data_path = f"{destination_path.name}.data"

        external_tensors = _external_data.convert_tensors_to_external(
            tensors,  # type: ignore[arg-type]
            base_dir,
            data_path,
        )

        # Replace the initializer values with external tensors and save the model
        for initializer, external_tensor in zip(initializer_values, external_tensors):
            initializer.const_value = external_tensor
        ir.save(model, model_path)

        # Restore the original initializer values so the model is unchanged
        for initializer, tensor in zip(initializer_values, tensors):
            initializer.const_value = tensor

    else:
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
