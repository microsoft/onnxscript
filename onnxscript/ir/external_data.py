# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""External data related utilities."""

from __future__ import annotations

__all__ = [
    "set_base_dir",
    "unload_from_model",
    "load_to_model",
    "convert_tensors_to_external",
    "convert_tensors_from_external",
]

import dataclasses
import logging
import os
from typing import Iterator, Sequence

from onnxscript.ir import _core, _enums, _protocols
from onnxscript.ir import traversal as _traversal
from onnxscript.ir._polyfill import zip

# Note: If needed in future, add these as parameters to the function calls
# align_offset: Offset will always be page aligned and alloction granularity aligned for mmap support. This is done by padding previous tensor data with zeros keeping same length. Tensor data will be aligned if > align_threshold
_ALIGN_OFFSET = True
# align_threshold: Alignment threshold for size of data. Having a low threshold will waste file space for small initializers. Only when tensor's data is > the page_align_threshold it will be force aligned.
_ALIGN_THRESHOLD = 1048576  # 1MB
# allocation_granularity: The allocation Granularity for mmap() support. Typically 64KB for Windows & 4KB for other OSes.
_ALLOCATION_GRANULARITY = 65536  # 64KB


logger = logging.getLogger(__name__)


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
    for node in _traversal.RecursiveGraphIterator(graph):
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


def _external_tensor_to_memory_tensor(
    tensor: _protocols.TensorProtocol,
) -> _protocols.TensorProtocol:
    """Convert an external tensor to an in memory tensor.

    Args:
        tensor: An external tensor to load.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        An ir.Tensor object with the data loaded into memory.
    """
    if not isinstance(tensor, _core.ExternalTensor):
        raise TypeError(f"Expected ExternalTensor, got {type(tensor)}")
    # Copy the data as the .numpy() call references data from a file whose data is eventually modified
    tensor_data = tensor.numpy().copy()
    tensor.release()
    return _core.Tensor(tensor_data, name=tensor.name, dtype=tensor.dtype)


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


def _write_external_data(
    tensors: Sequence[_protocols.TensorProtocol],
    external_data_infos: Sequence[_ExternalDataInfo],
    file_path: str | os.PathLike,
) -> None:
    """Write tensor data to an external file according to information stored in ExternalDataInfo objects.

    Args:
        tensors: Tensors to be written as external data.
        external_data_infos: External data information stored for each tensor to be written as external data.
        file_path: Location to which external data is to be stored.
    """
    assert len(tensors) == len(external_data_infos), (
        "Number of tensors and external data infos should match"
    )
    with open(file_path, "wb") as data_file:
        for tensor, tensor_info in zip(tensors, external_data_infos, strict=True):
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


def _create_external_tensor(
    tensor: _protocols.TensorProtocol,
    external_data_info: _ExternalDataInfo,
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
) -> _core.ExternalTensor:
    """Create external tensors from external data information.

    Args:
        tensor: Tensor to be converted to external tensor.
        external_data_info: External data information stored for the tensor to be written as external data.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        External tensor created from the information.
    """
    return _core.ExternalTensor(
        os.path.normpath(relative_path),
        external_data_info.offset,
        external_data_info.length,
        tensor.dtype,  # type: ignore[arg-type]
        shape=tensor.shape,  # type: ignore[arg-type]
        name=tensor.name,  # type: ignore[arg-type]
        base_dir=os.path.normpath(base_dir),
    )


def convert_tensors_from_external(
    tensors: Sequence[_protocols.TensorProtocol],
) -> list[_protocols.TensorProtocol]:
    """Convert a sequence of external tensors to in-memory tensors.

    Args:
        tensors: External tensors to be converted to in-memory tensors.

    Returns:
        A list of in-memory tensors derived from a list of external tensors.
    """
    return [_external_tensor_to_memory_tensor(tensor) for tensor in tensors]


def convert_tensors_to_external(
    tensors: Sequence[_protocols.TensorProtocol],
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
) -> list[_core.ExternalTensor]:
    """Convert a sequence of any TensorProtocol tensors to external tensors.

    Existing external tensors are loaded to memory if they are referring to the
    same file path as the destination path.

    Args:
        tensors: Tensors to be converted to external tensors. They can be external tensors themselves.
        base_dir: Path of base directory.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.

    Returns:
        A list of external tensors derived from a list of input tensors. The order
        should match the input tensor order.
    """
    path = os.path.join(base_dir, relative_path)

    # Check if output path exists. Load pre-existing external data if it does.
    if os.path.exists(path):
        # Check if any tensor provided is using the destination file
        new_tensors = []
        for tensor in tensors:
            if (
                isinstance(tensor, _core.ExternalTensor)
                and os.path.exists(tensor.path)
                and os.path.samefile(path, tensor.path)
            ):
                # FIXME(shubhambhokare1): If there is a non-initializer tensor that
                # is referring to this file, that tensor is now invalid.
                # This is a special case we are ok not handling right now.
                new_tensors.append(_external_tensor_to_memory_tensor(tensor))
                # Mark the original external tensor as invalid because it is now pointing
                # to a file that is going to be overwritten.
                tensor.invalidate()
                logger.warning(
                    "External tensor %s is referring to the same file as the destination path. "
                    "It has been invalidated because the data file is changed. To avoid this, "
                    "save the external data to a different path or load the newly saved model back "
                    "with ir.load().",
                    tensor,
                )
            else:
                new_tensors.append(tensor)
        tensors = new_tensors

    external_data_infos: list[_ExternalDataInfo] = []
    # Sort all tensors based on tensor sizes, in order to avoid unnecessary alignment.
    # All the smaller tensors are written earlier and alignment is performed for the larger tensors.
    sorted_indices = sorted(range(len(tensors)), key=lambda i: tensors[i].nbytes)
    sorted_tensors = [tensors[i] for i in sorted_indices]

    # Compute external data information for each tensor and write to disk
    current_offset = 0
    for tensor in sorted_tensors:
        external_info = _compute_external_data_info(tensor, current_offset)
        external_data_infos.append(external_info)
        current_offset = external_info.offset + external_info.length
    _write_external_data(sorted_tensors, external_data_infos, path)

    # Create external tensor objects
    external_tensors: list[_core.ExternalTensor] = [
        _create_external_tensor(tensor, external_info, base_dir, relative_path)
        for tensor, external_info in zip(sorted_tensors, external_data_infos, strict=True)
    ]

    # Sort external_tensors based on original key order. So that it can match the input tensor order
    external_tensors = [
        external_tensors[i]
        for i in sorted(range(len(external_tensors)), key=lambda i: sorted_indices[i])
    ]

    return external_tensors


def load_to_model(model: _core.Model) -> _core.Model:
    """Convert all external model initializers to memory tensors in-place.

    Args:
        model: Model to process.
    """
    # TODO(justinchuby): Load attributes and initializers in subgraphs
    values_to_convert = []
    for value in model.graph.initializers.values():
        if value.const_value is None:
            # Filter out the uninitialized initializer values
            continue
        if isinstance(value.const_value, _core.ExternalTensor):
            values_to_convert.append(value)
    loaded_tensors = convert_tensors_from_external(
        [v.const_value for v in values_to_convert]  # type: ignore[misc]
    )
    for value, tensor in zip(values_to_convert, loaded_tensors, strict=True):
        value.const_value = tensor

    # Return the model because we may change the implementation to an out of place one
    # to keep the input unchanged
    return model


def unload_from_model(
    model: _core.Model,
    base_dir: str | os.PathLike,
    relative_path: str | os.PathLike,
    *,
    size_threshold_bytes: int = 0,
) -> _core.Model:
    """Convert all initializers equal or above size_threshold_bytes to external tensors in-place and save data to a single data file.

    It should only replace the initializers in the model with external tensors
    and not make any other modifications to the model.

    If any existing external tensor
    references the provided ``external_data`` path, it will be invalidated
    after the external data is overwritten. To obtain a valid model, use :func:`load`
    to load the newly saved model, or provide a different external data path that
    is not currently referenced by any tensors in the model.

    Args:
        model: Model to process.
        base_dir: Path the directory where the ONNX model file is.
        relative_path: Path to which external data is to be stored, relative to the ONNX file.
            E.g. "model.data"
        size_threshold_bytes: Save to external data if the tensor size in bytes is larger than this threshold.

    Returns:
        An ir.Model with all initializer data equal or above ``size_threshold_bytes``
        converted to external tensors.
    """
    # In-memory or external tensors, if equal to or above the threshold, should be converted to or re-saved as external tensors
    initializers_to_become_external = []
    # Existing external tensors, if below the threshold, should be loaded to memory
    initializers_to_load_to_memory = []
    for value in model.graph.initializers.values():
        if value.const_value is None:
            # Filter out the uninitialized initializer values
            continue
        if value.const_value.nbytes > size_threshold_bytes:
            initializers_to_become_external.append(value)
        elif isinstance(value.const_value, _core.ExternalTensor):
            initializers_to_load_to_memory.append(value)

    # Load to memory first, then convert to external tensors, because
    # the existing external tensors may be overwritten by the new external data
    memory_tensors = convert_tensors_from_external(
        [v.const_value for v in initializers_to_load_to_memory]  # type: ignore[misc]
    )
    external_tensors = convert_tensors_to_external(
        [v.const_value for v in initializers_to_become_external],  # type: ignore[misc]
        base_dir=base_dir,
        relative_path=relative_path,
    )

    # Replace the initializer values with external tensors and save the model
    for value, external_tensor in zip(
        initializers_to_become_external, external_tensors, strict=True
    ):
        value.const_value = external_tensor
    for value, memory_tensor in zip(
        initializers_to_load_to_memory, memory_tensors, strict=True
    ):
        value.const_value = memory_tensor

    # Return the model because we may change the implementation to an out of place one
    # to keep the input unchanged
    return model
