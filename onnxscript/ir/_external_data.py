# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""External data related utilities."""

from __future__ import annotations

__all__ = ["set_base_dir"]

import os
from typing import Iterator

from onnxscript.ir import _core, _enums, _protocols, traversal


def _all_tensors(
    graph: _core.Graph | _core.GraphView, include_attributes: bool = False
) -> Iterator[_protocols.TensorProtocol]:
    """Iterate over all tensors in the graph.

    Args:
        graph: The graph to traverse tensors on.
        include_attributes: Whether to include tensors in attributes.

    Yields:
        Tensors in the graph.
    """
    # Yield all tensors in initializers
    for value in graph.initializers.values():
        if value.const_value is not None:
            yield value.const_value
    if not include_attributes:
        return
    # Look at constant attributes in nodes
    for node in traversal.RecursiveGraphIterator(graph):
        for attr in node.attributes.values():
            if isinstance(attr, _core.RefAttr):
                continue
            if attr.type == _enums.AttributeType.TENSOR and attr.value is not None:
                yield attr.value
            elif attr.type == _enums.AttributeType.TENSORS and attr.value is not None:
                yield from attr.value


def set_base_dir(graph: _core.Graph | _core.GraphView, base_dir: str | os.PathLike) -> None:
    """Set the base directory for external data in a graph.

    Args:
        graph: The graph to traverse tensors on.
        base_dir: The base directory. This is the directory where the ONNX file is.
    """
    for tensor in _all_tensors(graph, include_attributes=True):
        if isinstance(tensor, _core.ExternalTensor):
            tensor.base_dir = base_dir
"""Pass to save tensor data as external tensors."""

import os

from onnxscript.ir import _core


class ExternalDataInfo:
    def __init__(self, name, offset, length):
        self.name = name
        self.offset = offset
        self.length = length


def save_external_data(
    initializers,
    file_path,
    allocation_granularity: int = 65536,  # 64KB
):
    # Store external data such as name, offset and size
    external_data_info = dict()

    # Create file if it doesn't exist
    if not os.path.isfile(file_path):
        with open(file_path, "ab"):
            pass

    with open(file_path, "r+b") as data_file:
        current_offset = 0
        for i_name, i_value in initializers.items():
            tensor_val = i_value
            if isinstance(tensor_val, _core.Value):
                tensor_val = i_value.const_value()  # pylint: disable=protected-access
            assert isinstance(tensor_val, _core.Tensor)
            raw_data = tensor_val.tobytes()
            tensor_size = tensor_val.size
            # Convert each initializer to core.ExternalTensor
            # Align tensors
            alignment_factor = max(4096, allocation_granularity)
            current_offset = (
                (current_offset + alignment_factor - 1) // alignment_factor * alignment_factor
            )

            data_file.seek(0, 2)
            if current_offset is not None:
                # Pad file to required offset if needed
                file_size = data_file.tell()
                if current_offset > file_size:
                    data_file.write(b"\0" * (current_offset - file_size))
                data_file.seek(current_offset)
            data_file.write(raw_data)

            # Store tensor external data
            external_data_info[i_value.name] = ExternalDataInfo(
                i_name,
                current_offset,
                tensor_size,
            )

            # Update offset
            current_offset += tensor_size
    return external_data_info


def convert_model_to_external_data(model: _core.Model, base_path: str, file_path: str = ""):
    file_path = os.path.join(base_path, file_path)
    external_data_info = save_external_data(model.graph.initializers, file_path)

    # Convert initializers to ExternalTensors
    for i_name, i_value in model.graph.initializers.items():
        assert i_name in external_data_info
        tensor_info = external_data_info[i_name]
        new_external_tensor = _core.ExternalTensor(
            file_path,
            tensor_info.offset,
            tensor_info.length,
            i_value.dtype,  # type: ignore[arg-type]
            shape=i_value.shape,  # type: ignore[arg-type]
            name=i_value.name,  # type: ignore[arg-type]
        )
        model.graph.initializers[i_name] = new_external_tensor  # type: ignore[assignment]
    return model
