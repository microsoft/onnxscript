# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load and save ONNX models."""

from __future__ import annotations

__all__ = ["load", "save"]

import os
from typing import Iterator

import onnx

from onnxscript.ir import _core, _enums, _protocols, serde, traversal


def _all_tensors(
    graph: _core.Graph | _core.GraphView, include_constants: bool = False
) -> Iterator[_protocols.TensorProtocol]:
    """Iterate over all tensors in the graph."""

    # Yield all tensors in initializers
    for value in graph.initializers.values():
        if value.const_value is not None:
            yield value.const_value
    if not include_constants:
        return
    # Look at constant attributes in nodes
    for node in traversal.RecursiveGraphIterator(graph):
        for attr in node.attributes.values():
            if isinstance(attr, _core.RefAttr):
                continue
            if attr.type == _enums.AttributeType.TENSOR and attr.value is not None:
                yield attr.value
            elif attr.type == _enums.AttributeType.TENSORS and attr.value is not None:
                for value in attr.value:
                    yield value


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
    for tensor in _all_tensors(model.graph, include_constants=True):
        if isinstance(tensor, _core.ExternalTensor):
            tensor.base_dir = base_dir
    return model


def save(model: _core.Model, path: str | os.PathLike, format: str | None = None) -> None:
    """Save an ONNX model to a file.

    Args:
        model: The model to save.
        path: The path to save the model to.
        format: The format of the file (e.g. protobuf, textproto, json, etc.).
            If None, the format is inferred from the file extension.
    """
    proto = serde.serialize_model(model)
    onnx.save(proto, path, format=format)
