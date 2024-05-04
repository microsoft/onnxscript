"""Numpy utilities for non-native type operation."""
# TODO(justinchuby): Upstream the logic to onnx

from __future__ import annotations

import typing
from typing import Sequence

import numpy as np

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def pack_int4(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to flatten, packed int4/uint4. Elements must be in the correct range."""
    # Create a 1D copy
    array_flat = array.ravel().astype(np.uint8)
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]  # type: ignore[return-type]


def unpack_uint4(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def _int4_to_int8(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.int8]:
    """Extend 4-bit signed integer to 8-bit signed integer."""
    return np.where((x >> 3) == 0, x, x | 0xF0).astype(np.int8)


def unpack_int4(data: npt.NDArray[np.uint8], dims: Sequence[int]) -> npt.NDArray[np.int8]:
    """Convert a packed (signed) int4 array to unpacked int4 array represented as int8.

    The sign bit is extended to the most significant bit of the int8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8 reshaped to dims.
    """
    unpacked = unpack_uint4(data, dims)
    return _int4_to_int8(unpacked)


def float32_to_bfloat16(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """Convert a numpy array to uint16 representation of bfloat16."""
    bfloat16_array = array.astype(np.float32).view(np.uint32)
    # Drop bottom 16-bits
    # Round remaining bits using round-to-nearest-even
    rounded = bfloat16_array >> 16
    rounded &= 1
    rounded += 0x7FFF
    bfloat16_array += rounded  # type: ignore[arg-type]
    bfloat16_array >>= 16
    # NaN requires at least 1 significant bit set
    bfloat16_array[np.isnan(array)] = 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000
    return bfloat16_array.astype(np.uint16)
