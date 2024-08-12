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

from onnxscript.ir._core import Model, Tensor, ExternalTensor


class ExternalDataInfo:
    def __init__(self, external_tensor: ExternalTensor):
        self.offset = external_tensor._offset
        self.length = external_tensor._length


def save_external_data(raw_data, external_tensor: ExternalTensor, path: str):
    external_data = ExternalDataInfo(external_tensor)
    external_data_file_path = os.path.join(path, "tens_data")

    with open("tens_data_1", "w+b") as data_file:
        data_file.seek(0, 2)
        if external_data.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if external_data.offset > file_size:
                data_file.write(b"\0" * (external_data.offset - file_size))

            data_file.seek(external_data.offset)
        #offset = data_file.tell()
        data_file.write(raw_data)
        #set_external_data(tensor, info.location, offset, data_file.tell() - offset)


def _convert_initializers_to_external_tensor(
    initializers,
    path,
    allocation_granularity: int = 65536, #64KB
):
    current_offset = 0
    for i_name, i_value in initializers.items():
        print(i_name)

        raw_data = i_value._raw
        tensor_size = raw_data.size
        # Convert each initializer to core.ExternalTensor
        # Align tensors
        alignment_factor = max(4096, allocation_granularity)
        current_offset = (current_offset + alignment_factor - 1) // alignment_factor * alignment_factor

        new_external_tensor = ExternalTensor(
            path,
            current_offset,
            tensor_size,
            i_value.dtype,
            shape=i_value.shape,
            name=i_value.name,
        )
        initializers[i_name] = new_external_tensor

        # Write data to file
        save_external_data(raw_data, new_external_tensor, path)

        # Update offset
        current_offset += tensor_size
    return initializers



def _get_all_initializers(model: Model):
    graph = model.graph
    return graph.initializers

def convert_model_to_external_data(model: Model, path):
    initializers = _get_all_initializers(model)
    print(initializers)
    new_initializers = _convert_initializers_to_external_tensor(initializers, path)
    print(new_initializers)

