# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load and save ONNX models."""

from __future__ import annotations

__all__ = ["load", "save"]

import os

import onnx

from onnxscript.ir import _core, serde
from onnxscript.ir import external_data as _external_data
from onnxscript.ir._polyfill import zip


def load(path: str | os.PathLike, format: str | None = None) -> _core.Model:
    """Load an ONNX model from a file.

    Args:
        path: The path to the ONNX file.
        format: The format of the file (e.g. protobuf, textproto, json, etc.).
            If None, the format is inferred from the file extension.

    Returns:
        The loaded model.
    """
    # Do not use ONNX to load external data because the IR handles external data
    # by doing memory mapping directly.
    proto = onnx.load(path, format=format, load_external_data=False)
    model = serde.deserialize_model(proto)
    base_dir = os.path.dirname(path)
    # Set the base directory for external data to the directory of the ONNX file
    # so that relative paths are resolved correctly.
    _external_data.set_base_dir(model.graph, base_dir)
    return model


def save(
    model: _core.Model,
    path: str | os.PathLike,
    format: str | None = None,
    external_data: str | os.PathLike | None = None,
    size_threshold_bytes: int = 256,
) -> None:
    """Save an ONNX model to a file.

    The model remains unchanged after the call. If any existing external tensor
    references the provided ``external_data`` path, it will be invalidated
    after the external data is overwritten. To obtain a valid model, use :func:`load`
    to load the newly saved model, or provide a different external data path that
    is not currently referenced by any tensors in the model.

    Args:
        model: The model to save.
        path: The path to save the model to. E.g. "model.onnx".
        format: The format of the file (e.g. ``protobuf``, ``textproto``, ``json``, etc.).
            If None, the format is inferred from the file extension.
        external_data: The relative path to save external data to. When specified,
            all initializers in the model will be converted to external data and
            saved to the specified directory. If None, all tensors will be saved unmodified.
            That is, if a tensor in the model is already external, it will be saved
            with the same external information; if the tensor is not external,
            it will be serialized in the ONNX Proto message.
        size_threshold_bytes: Save to external data if the tensor size in bytes is larger than this threshold.
            Effective only when ``external_data`` is set.

    Raises:
        ValueError: If the external data path is an absolute path.
    """
    if external_data is not None:
        if os.path.isabs(external_data):
            raise ValueError(
                f"The external data path must be relative to the ONNX file path, not '{external_data}'."
            )
        base_dir = os.path.dirname(path)

        # Store the original initializer values so they can be restored if modify_model=False
        initializer_values = tuple(model.graph.initializers.values())
        tensors = [v.const_value for v in initializer_values]

        try:
            model = _external_data.unload_from_model(
                model, base_dir, external_data, size_threshold_bytes=size_threshold_bytes
            )
            proto = serde.serialize_model(model)
            onnx.save(proto, path, format=format)

        finally:
            # Restore the original initializer values so the model is unchanged
            for initializer, tensor in zip(initializer_values, tensors, strict=True):
                initializer.const_value = tensor

    else:
        proto = serde.serialize_model(model)
        onnx.save(proto, path, format=format)
