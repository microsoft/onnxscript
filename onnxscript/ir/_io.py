# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load and save ONNX models."""

from __future__ import annotations

import os

import onnx

from onnxscript.ir import _core, serde


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
    _external_data.set_base_dir(model, base_dir)
    return model


def save(model: _core.Model, path: str | os.PathLike, format: str | None = None) -> None:
    """Save an ONNX model to a file.

    Args:
        model: The model to save.
        path: The path to save the model to.
        format: The format of the file (e.g. protobuf, textproto, json, etc.).
            If None, the format is inferred from the file extension.
    """
    onnx_model = serde.serialize_model(model)
    onnx.save(onnx_model, path, format=format)
