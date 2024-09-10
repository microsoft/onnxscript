# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""External data related utilities."""

from __future__ import annotations

__all__ = ["set_base_dir"]

import dataclasses
import os
from typing import Iterator, Sequence

from onnxscript.ir import _core, _enums, _protocols, traversal

# Note: If needed in future, add these as parameters to the function calls
# align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
_ALIGN_OFFSET = True
# align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
_ALIGN_THRESHOLD = 1048576  # 1MB
# allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.
_ALLOCATION_GRANULARITY = 65536  # 64KB


@dataclasses.dataclass
class _ExternalDataInfo:
    """
    A class that stores information about a tensor that is to be stored as external data.

    Attributes:
        name: The name of the tensor that is to be stored as external data.
        offset: The offset is used to determine where exactly in the file the external data is written to.
        length: Stores the size of the tensor.
    """

    name: str | None
    offset: int
    length: int


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


def _load_external_data_file(
    tensors: Sequence[_protocols.TensorProtocol],
    base_path: str | os.PathLike,
    relative_path: str | os.PathLike,
) -> list[_protocols.TensorProtocol]:
    """Load all external data that is at relative_path into memory for the provided model.

    Args:
        tensors: Tensors to be converted to external tensors. They can be external tensors themselves.
        base_path: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        A list of ir.Tensor values.
    """
    updated_tensors: list[_protocols.TensorProtocol] = []
    for tensor in tensors:
        if isinstance(tensor, _core.ExternalTensor):
            external_tensor = tensor
            if os.path.samefile(tensor.path, os.path.join(base_path, relative_path)):
                # Copy the data as the .numpy() call references data from a file whose data is eventually modified
                tensor_data = external_tensor.numpy().copy()
                external_tensor.release()
                tensor = _core.Tensor(
                    tensor_data, name=external_tensor.name, dtype=external_tensor.dtype
                )
        updated_tensors.append(tensor)
    return updated_tensors


def _compute_new_offset(
    current_offset: int,
    tensor_size: int,
    align_offset: bool = _ALIGN_OFFSET,
    align_threshold: int = _ALIGN_THRESHOLD,
    allocation_granularity: int = _ALLOCATION_GRANULARITY,
) -> int:
    """Compute the offset to align the tensor data based on the current offset.

    Args:
        current_offset: Current location in the file at which tensor data will be written to.
        tensor_size: Size of the tensor data to be written to file.
        align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
        align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
        allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.

    Returns:
        The updated offset value.
    """
    if align_offset and tensor_size > align_threshold:
        alignment_factor = max(4096, allocation_granularity)
        # Align to the next page or alloc granularity
        return (current_offset + alignment_factor - 1) // alignment_factor * alignment_factor
    return current_offset


def _compute_external_data_info(
    tensor: _protocols.TensorProtocol,
    current_offset: int,
) -> _ExternalDataInfo:
    """Capture information about a tensor that is to be stored as external data."""
    tensor_size = tensor.nbytes
    # Calculate updated offset and align tensors
    current_offset = _compute_new_offset(current_offset, tensor_size)
    # Store offset and tensor size as ExternalDataInfo
    external_data_info = _ExternalDataInfo(
        tensor.name,
        current_offset,
        tensor_size,
    )
    return external_data_info


def _save_external_data(
    external_data_info: list[tuple[_protocols.TensorProtocol, _ExternalDataInfo]],
    file_path: str | os.PathLike,
) -> None:
    """Write tensor data to an external file according to information stored in ExternalDataInfo objects.

    Args:
        external_data_info: A collection of external data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
    """
    with open(file_path, "wb") as data_file:
        for tensor, tensor_info in external_data_info:
            current_offset = tensor_info.offset
            assert tensor is not None
            raw_data = tensor.tobytes()
            if isinstance(tensor, _core.ExternalTensor):
                tensor.release()
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if current_offset > file_size:
                data_file.write(b"\0" * (current_offset - file_size))
            data_file.write(raw_data)


def _convert_as_external_tensors(
    external_data_info: list[tuple[_protocols.TensorProtocol, _ExternalDataInfo]],
    base_path: str | os.PathLike,
    relative_path: str | os.PathLike,
) -> list[_core.ExternalTensor]:
    """Convert the tensors (stored within the values) written as external data to _core.ExternalTensor types.

    Args:
        external_data_info: A collection of external data information stored for each tensor to be written as external data.
        base_path: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        A list of external tensors.
    """
    external_tensors: list[_core.ExternalTensor] = []
    for tensor, tensor_info in external_data_info:
        assert tensor is not None
        external_tensor = _core.ExternalTensor(
            os.path.normpath(relative_path),
            tensor_info.offset,
            tensor_info.length,
            tensor.dtype,  # type: ignore[arg-type]
            shape=tensor.shape,  # type: ignore[arg-type]
            name=tensor.name,  # type: ignore[arg-type]
            base_dir=os.path.normpath(base_path),
        )
        external_tensors.append(external_tensor)
    return external_tensors


def convert_tensors_to_external(
    tensors: Sequence[_protocols.TensorProtocol],
    base_path: str | os.PathLike,
    relative_path: str | os.PathLike,
    load_external_to_memory: bool = False,
) -> list[_core.ExternalTensor]:
    """Convert a sequence of any TensorProtocol tensors to external tensors.

    Args:
        tensors: Tensors to be converted to external tensors. They can be external tensors themselves.
        base_path: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
        load_external_to_memory: If set to true, loads external tensors present in the same file path as destination path to memory.

    Returns:
        A list of external tensors derived from a list of input tensors.
    """
    path = os.path.join(base_path, relative_path)
    # Check if file path is valid, and create subsequent subdirectories within the path if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_file_created = False
    # Check if file exists. Load pre-existing external data if it does.
    if os.path.exists(path):
        # Check if any tensor in the model is using the destination file
        file_used = False
        for tensor in tensors:
            if isinstance(tensor, _core.ExternalTensor) and os.path.samefile(
                path, tensor.path
            ):
                # FIXME(shubhambhokare1): If there is a non-initializer tensor that is referring to this file, that tensor is now invalid. This is a special case we are ok not handling right now.
                file_used = True
        if file_used:
            if load_external_to_memory:
                tensors = _load_external_data_file(tensors, base_path, relative_path)
            else:
                tmp_path = os.path.join(base_path, "tmp")
                os.makedirs(tmp_path, exist_ok=True)
                # If exisiting external tensors are not loaded to memory, copy the external data to a temporary location
                os.rename(path, os.path.join(tmp_path, relative_path))
                tmp_file_created = True
                for tensor in tensors:
                    if (
                        isinstance(tensor, _core.ExternalTensor)
                        and tensor.location == relative_path
                    ):
                        tensor.base_dir = tmp_path

    external_data_info: list[tuple[_protocols.TensorProtocol, _ExternalDataInfo]] = []
    # Sort all tensors based on tensor sizes, in order to avoid unneccesarry alignment.
    # All the smaller tensors are written earlier and alignment is performed for the larger tensors.
    sorted_indices = sorted(range(len(tensors)), key=lambda i: tensors[i].nbytes)
    sorted_tensors = [tensors[i] for i in sorted_indices]

    current_offset = 0
    for tensor in sorted_tensors:
        tensor_info = _compute_external_data_info(tensor, current_offset)
        external_data_info.append((tensor, tensor_info))
        current_offset = tensor_info.offset + tensor_info.length
    _save_external_data(external_data_info, path)

    # Convert initializers to ExternalTensors
    external_tensors = _convert_as_external_tensors(
        external_data_info, base_path, relative_path
    )
    # Sort external_tensors based on original key order
    external_tensors = [
        external_tensors[i]
        for i in sorted(range(len(external_tensors)), key=lambda i: sorted_indices[i])
    ]

    # Clean-up temporary file if it is created
    tmp_path = os.path.join(base_path, "tmp", relative_path)
    if os.path.exists(tmp_path) and tmp_file_created:
        os.remove(tmp_path)

    return external_tensors


def to_external_data(
    model: _core.Model,
    base_path: str | os.PathLike,
    relative_path: str | os.PathLike,
    load_external_to_memory: bool = False,
) -> _core.Model:
    """Set all tensors with raw data as external data.

    Args:
        model: Model to process.
        base_path: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
        load_external_to_memory: If set to true, loads external tensors present in the same file path as destination path to memory. Otherwise, the external tensors are appended to file.

    Returns:
        An ir.Model with all tensors with raw data converted to external tensors.
    """

    # Get all the tensors in the graph which are to be stored as external data.
    # Iterate through all the tensors, and extract the external data information such as
    # name, offset and length.
    # TODO: Currently attributes not handled, eventually try to use _all_tensors to include attrs
    tensors: list[_protocols.TensorProtocol] = []
    for value in model.graph.initializers.values():
        if value.const_value is not None:
            tensors.append(value.const_value)

    external_tensors = convert_tensors_to_external(
        tensors,
        base_path,
        relative_path,
        load_external_to_memory=load_external_to_memory,
    )

    for value, external_tensor in zip(model.graph.initializers.values(), external_tensors):
        value.const_value = external_tensor
    return model
