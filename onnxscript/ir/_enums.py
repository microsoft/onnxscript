# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX IR enums that matches the ONNX spec."""

from __future__ import annotations

import enum

import numpy as np


class AttributeType(enum.IntEnum):
    """Enum for the types of ONNX attributes."""

    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    TYPE_PROTO = 13
    TYPE_PROTOS = 14

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class DataType(enum.IntEnum):
    """Enum for the data types of ONNX tensors, defined in ``onnx.TensorProto``."""

    # NOTE: Naming: It is tempting to use shorter and more modern names like f32, i64,
    # but we should stick to the names used in the ONNX spec for consistency.
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        """Returns the ONNX data type for the numpy dtype.

        Raises:
            TypeError: If the data type is not supported by ONNX.
        """
        if dtype not in _NP_TYPE_TO_DATA_TYPE:
            raise TypeError(f"Unsupported numpy data type: {dtype}")
        return cls(_NP_TYPE_TO_DATA_TYPE[dtype])

    @property
    def itemsize(self) -> float:
        """Returns the size of the data type in bytes."""
        return _ITEMSIZE_MAP[self]

    def numpy(self) -> np.dtype:
        """Returns the numpy dtype for the ONNX data type.

        Raises:
            TypeError: If the data type is not supported by numpy.
        """
        if self not in _DATA_TYPE_TO_NP_TYPE:
            raise TypeError(f"Numpy does not support ONNX data type: {self}")
        return _DATA_TYPE_TO_NP_TYPE[self]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


_ITEMSIZE_MAP = {
    DataType.FLOAT: 4,
    DataType.UINT8: 1,
    DataType.INT8: 1,
    DataType.UINT16: 2,
    DataType.INT16: 2,
    DataType.INT32: 4,
    DataType.INT64: 8,
    DataType.STRING: 1,
    DataType.BOOL: 1,
    DataType.FLOAT16: 2,
    DataType.DOUBLE: 8,
    DataType.UINT32: 4,
    DataType.UINT64: 8,
    DataType.COMPLEX64: 8,
    DataType.COMPLEX128: 16,
    DataType.BFLOAT16: 2,
    DataType.FLOAT8E4M3FN: 1,
    DataType.FLOAT8E4M3FNUZ: 1,
    DataType.FLOAT8E5M2: 1,
    DataType.FLOAT8E5M2FNUZ: 1,
    DataType.UINT4: 0.5,
    DataType.INT4: 0.5,
}


_NP_TYPE_TO_DATA_TYPE = {
    np.dtype("bool"): DataType.BOOL,
    np.dtype("complex128"): DataType.COMPLEX128,
    np.dtype("complex64"): DataType.COMPLEX64,
    np.dtype("float16"): DataType.FLOAT16,
    np.dtype("float32"): DataType.FLOAT,
    np.dtype("float64"): DataType.DOUBLE,
    np.dtype("int16"): DataType.INT16,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int64"): DataType.INT64,
    np.dtype("int8"): DataType.INT8,
    np.dtype("object"): DataType.STRING,
    np.dtype("uint16"): DataType.UINT16,
    np.dtype("uint32"): DataType.UINT32,
    np.dtype("uint64"): DataType.UINT64,
    np.dtype("uint8"): DataType.UINT8,
}

# ONNX DataType to Numpy dtype. This mapping does not capture ONNX data
# types that are not supported by numpy.
_DATA_TYPE_TO_NP_TYPE = {v: k for k, v in _NP_TYPE_TO_DATA_TYPE.items()}
_DATA_TYPE_TO_NP_TYPE.update(
    {
        DataType.FLOAT8E4M3FN: np.dtype("uint8"),
        DataType.FLOAT8E4M3FNUZ: np.dtype("uint8"),
        DataType.FLOAT8E5M2: np.dtype("uint8"),
        DataType.FLOAT8E5M2FNUZ: np.dtype("uint8"),
        DataType.UINT4: np.dtype("uint8"),
        DataType.INT4: np.dtype("int8"),
        DataType.BFLOAT16: np.dtype("uint16"),
    }
)
