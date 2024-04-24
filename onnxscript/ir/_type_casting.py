"""Numpy utilities for subbyte operation."""
# TODO(justinchuby): Upstream the logic to onnx

from __future__ import annotations

import typing

import numpy as np
import onnx

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def pack_int4(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to flatten, packed int4/uint4. Elements must be in the correct range."""
    # Create a 1D copy
    array_flat = array.ravel().astype(np.uint8)
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        size = size + 1
        array_flat.resize([size])
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]  # type: ignore[return-type]


def float32_to_bfloat16(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """Convert a numpy array to uint16 representation of bfloat16."""
    bfloat16_array = array.astype(np.float32).view(np.uint32)
    # NaN requires at least 1 significand bit set
    bfloat16_array[np.isnan(array)] = 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000
    # Drop bottom 16-bits
    # Round remaining bits using round-to-nearest-even
    rounded = bfloat16_array >> 16
    rounded &= 1
    rounded += 0x7FFF
    bfloat16_array += rounded
    bfloat16_array >>= 16
    return bfloat16_array.astype(np.uint16)


def float32_to_float8e5m2(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to uint8 representation of float8e5m2."""
    func = np.frompyfunc(onnx.helper.float32_to_float8e5m2, 1, 1)
    return func(array)


def float32_to_float8e5m2fnuz(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to uint8 representation of float8e5m2fnuz."""
    func = np.frompyfunc(
        lambda x: onnx.helper.float32_to_float8e5m2(x, fn=True, uz=True), 1, 1
    )
    return func(array)


def float32_to_float8e4m3fn(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to uint8 representation of float8e4m3fn."""
    func = np.frompyfunc(onnx.helper.float32_to_float8e4m3, 1, 1)
    return func(array)


def float32_to_float8e4m3fnuz(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to uint8 representation of float8e4m3fnuz."""
    func = np.frompyfunc(lambda x: onnx.helper.float32_to_float8e4m3(x, uz=True), 1, 1)
    return func(array)
