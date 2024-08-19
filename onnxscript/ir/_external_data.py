# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""External data related utilities."""

from __future__ import annotations

__all__ = ["set_base_dir"]

import os
import pathlib
from typing import Iterator

import numpy as np

from onnxscript.ir import _core, _enums, _protocols, traversal


class ExternalDataInfo:
    """
    A class that stores information about a tensor that is to be stored as external data.

    Args:
        name: The name of the tensor that is to be stored as external data.
        offset: The offset is used to determine where exactly in the file the external data is written to.
        length: Stores the size of the tensor.
    """

    def __init__(self, name: str, offset: int, length: int):
        self.name = name
        self.offset = offset
        self.length = length


def _all_tensors(
    graph: _core.Graph | _core.GraphView, include_attributes: bool = False
) -> Iterator[
    tuple[_protocols.TensorProtocol, _protocols.ValueProtocol | _protocols.AttributeProtocol]
]:
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


# Loading external data


def check_external_data_file(model: _core.Model, file_path: str | os.PathLike) -> None:
    """
    Check if file to which external data is to be written to is empty.
    If file already consists of external data, load external data.

    Args:
        model: Model to be converted.
        file_path: Location to which external data is to be stored.
    """
    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        return
    # If file is not empty, load external data
    with open(file_path, "r+b") as data_file:
        for value in model.graph.initializers.values():
            if isinstance(value.const_value, _core.ExternalTensor):
                external_tensor = value.const_value
                external_tensor_path = (
                    pathlib.Path(external_tensor._base_dir) / external_tensor._path
                )
                assert external_tensor_path == file_path
                data_file.seek(external_tensor.offset)

                # Read raw data from file based on offset and length
                # TODO: Convert this into a more robust function
                raw_data = data_file.read(external_tensor.length)
                raw_data = np.frombuffer(raw_data, dtype=external_tensor.dtype.numpy())
                raw_data = np.reshape(raw_data, newshape=external_tensor.shape.numpy())

                # Store raw data as a tensor and update value.const_value
                tensor = _core.Tensor(
                    raw_data, name=external_tensor.name, dtype=external_tensor.dtype
                )
                value.const_value = tensor
                model.graph.initializers[value.name] = value


# Converting model initializers to external data


def _compute_new_offset(
    current_offset: int,
    tensor_size: int,
    align_offset: bool = True,
    align_threshold: int = 1048576,  # 1MB,
    allocation_granularity: int = 65536,  # 64KB
) -> int:
    """
    Method to compute updated offset.

    Args:
        current_offset: Current location in the file at which tensor data will be written to.
        tensor_size: Size of the tensor data to be written to file.
        align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
        align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
        allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.
    """
    if align_offset and tensor_size > align_threshold:
        alignment_factor = max(4096, allocation_granularity)
        # Align to the next page or alloc granularity
        current_offset = (
            (current_offset + alignment_factor - 1) // alignment_factor * alignment_factor
        )
    return current_offset


def set_external_data(
    tensor: _protocols.TensorProtocol,
    current_offset: int,
    align_offset: bool = True,
    align_threshold: int = 1048576,  # 1MB,
    allocation_granularity: int = 65536,  # 64KB
) -> ExternalDataInfo:
    """
    Method to capture information about a tensor that is to be stored as external data.
    """
    tensor_size = tensor.nbytes
    # Calculate updated offset and align tensors
    current_offset = _compute_new_offset(
        current_offset,
        tensor_size,
        align_offset=align_offset,
        align_threshold=align_threshold,
        allocation_granularity=allocation_granularity,
    )
    # Store offset and tensor size as ExternalDataInfo
    external_data_info = ExternalDataInfo(
        tensor.name,
        current_offset,
        tensor_size,
    )
    return external_data_info


def save_external_data(
    external_data_info: list[tuple[_core.Value, ExternalDataInfo]],
    file_path: str | os.PathLike,
) -> None:
    """
    Writes tensor data to an external file according to information stored in ExternalDataInfo object.

    Args:
        external_data_info: A collection of external data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
    """
    with open(file_path, "w+b") as data_file:
        for value, tensor_info in external_data_info:
            current_offset = tensor_info.offset
            raw_data = value._const_value.tobytes()
            if current_offset is not None:
                # Pad file to required offset if needed
                file_size = data_file.tell()
                if current_offset > file_size:
                    data_file.write(b"\0" * (current_offset - file_size))
            data_file.write(raw_data)


def store_as_external_tensors(
    model: _core.Model,
    external_data_info: list[tuple[_core.Value, ExternalDataInfo]],
    file_path: str | os.PathLike,
) -> _core.Model:
    """
    Convert the tensors (stored within the values) written as external data to _core.ExternalTensor types.

    Args:
        model: Model to be converted.
        external_data_info: A collection of external data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
    """
    for value, tensor_info in external_data_info:
        tensor = value.const_value
        external_tensor = _core.ExternalTensor(
            file_path,
            tensor_info.offset,
            tensor_info.length,
            tensor.dtype,  # type: ignore[arg-type]
            shape=tensor.shape,  # type: ignore[arg-type]
            name=tensor.name,  # type: ignore[arg-type]
        )
        value.const_value = external_tensor
        assert value.name in model.graph.initializers
        model.graph.initializers[value.name] = value
    return model


def to_external_data(
    model: _core.Model,
    base_path: str | os.PathLike,
    file_path: str = "",
    align_offset: bool = True,
    align_threshold: int = 1048576,  # 1MB,
    allocation_granularity: int = 65536,  # 64KB
) -> _core.Model:
    """
    Call to set all tensors with raw data as external data.

    Args:
        model: Model to be converted.
        base_path: Path of base directory.
        file_path: Path to which external data is to be stored.
        align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
        align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
        allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.
    """
    file_path = pathlib.Path(base_path) / file_path
    # Check if file path is valid, and create subsequent subdirectories within the path if they don't exist
    try:
        file_path.mkdir(parents=True, exist_ok=True)
    except:
        pass
    # Check if file is empty. If not, load pre-existing external data.
    check_external_data_file(model, file_path)

    # Get all the tensors in the graph which are to be stored as external data.
    # Iterate through all the tensors, and extract the external data information such as
    # name, offset and length.
    # TODO: Currently attributes not handled, eventually try to use _all_tensors to include attrs
    tensors: list[tuple[_core.Value, _protocols.TensorProtocol]] = []
    for value in model.graph.initializers.values():
        if value.const_value is not None:
            tensors.append((value, value.const_value))
    external_data_info: list[tuple[_core.Value, ExternalDataInfo]] = []

    # Sort all tensors based on tensor sizes, in order to avoid unneccesarry alignment.
    # All the smaller tensors are written earlier and alignment is peformed for the larger tensors.
    tensors = sorted(tensors, key=lambda x: x[1].nbytes)

    current_offset = 0
    for value, tensor in tensors:
        tensor_info = set_external_data(
            tensor,
            current_offset,
            align_offset=align_offset,
            align_threshold=align_threshold,
            allocation_granularity=allocation_granularity,
        )
        external_data_info.append((value, tensor_info))
        current_offset = tensor_info.offset + tensor_info.length

    save_external_data(external_data_info, file_path)

    # Convert initializers to ExternalTensors
    model = store_as_external_tensors(model, external_data_info, file_path)
    return model
