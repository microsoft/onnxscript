# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ONNX IR enums that matches the ONNX spec."""

from __future__ import annotations

import enum

import ml_dtypes
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
    FLOAT4E2M1 = 23

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DataType:
        """Returns the ONNX data type for the numpy dtype.

        Raises:
            TypeError: If the data type is not supported by ONNX.
        """
        if dtype in _NP_TYPE_TO_DATA_TYPE:
            return cls(_NP_TYPE_TO_DATA_TYPE[dtype])

        if np.issubdtype(dtype, np.str_):
            return DataType.STRING

        # Special cases for handling custom dtypes defined in ONNX (as of onnx 1.18)
        # Ref: https://github.com/onnx/onnx/blob/2d42b6a60a52e925e57c422593e88cc51890f58a/onnx/_custom_element_types.py
        if hasattr(dtype, "names"):
            if dtype.names == ("bfloat16",):
                return DataType.BFLOAT16
            if dtype.names == ("e4m3fn",):
                return DataType.FLOAT8E4M3FN
            if dtype.names == ("e4m3fnuz",):
                return DataType.FLOAT8E4M3FNUZ
            if dtype.names == ("e5m2",):
                return DataType.FLOAT8E5M2
            if dtype.names == ("e5m2fnuz",):
                return DataType.FLOAT8E5M2FNUZ
            if dtype.names == ("uint4",):
                return DataType.UINT4
            if dtype.names == ("int4",):
                return DataType.INT4
            if dtype.names == ("float4e2m1",):
                return DataType.FLOAT4E2M1
        raise TypeError(f"Unsupported numpy data type: {dtype}")

    @classmethod
    def from_short_name(cls, short_name: str) -> DataType:
        """Returns the ONNX data type for the short name.

        Raises:
            TypeError: If the short name is not available for the data type.
        """
        if short_name not in _SHORT_NAME_TO_DATA_TYPE:
            raise TypeError(f"Unknown short name: {short_name}")
        return cls(_SHORT_NAME_TO_DATA_TYPE[short_name])

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

    def short_name(self) -> str:
        """Returns the short name of the data type.

        The short name is a string that is used to represent the data type in a more
        compact form. For example, the short name for `DataType.FLOAT` is "f32".
        To get the corresponding data type back, call ``from_short_name`` on a string.

        Naming reference: https://github.com/pytorch/pytorch/blob/4bead7b85ea4160243c74109e0ce9bb80686d016/torch/utils/_dtype_abbrs.py

        Raises:
            TypeError: If the short name is not available for the data type.
        """
        if self not in _DATA_TYPE_TO_SHORT_NAME:
            raise TypeError(f"Short name not available for ONNX data type: {self}")
        return _DATA_TYPE_TO_SHORT_NAME[self]

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
    DataType.FLOAT4E2M1: 0.5,
}


# We use ml_dtypes to support dtypes that are not in numpy.
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
    np.dtype(ml_dtypes.bfloat16): DataType.BFLOAT16,
    np.dtype(ml_dtypes.float8_e4m3fn): DataType.FLOAT8E4M3FN,
    np.dtype(ml_dtypes.float8_e4m3fnuz): DataType.FLOAT8E4M3FNUZ,
    np.dtype(ml_dtypes.float8_e5m2): DataType.FLOAT8E5M2,
    np.dtype(ml_dtypes.float8_e5m2fnuz): DataType.FLOAT8E5M2FNUZ,
    np.dtype(ml_dtypes.int4): DataType.INT4,
    np.dtype(ml_dtypes.uint4): DataType.UINT4,
}

# TODO(after min req for ml_dtypes>=0.5): Move this inside _NP_TYPE_TO_DATA_TYPE
_NP_TYPE_TO_DATA_TYPE.update(
    {np.dtype(ml_dtypes.float4_e2m1fn): DataType.FLOAT4E2M1}
    if hasattr(ml_dtypes, "float4_e2m1fn")
    else {}
)

# ONNX DataType to Numpy dtype.
_DATA_TYPE_TO_NP_TYPE = {v: k for k, v in _NP_TYPE_TO_DATA_TYPE.items()}

_DATA_TYPE_TO_SHORT_NAME = {
    DataType.UNDEFINED: "undefined",
    DataType.BFLOAT16: "bf16",
    DataType.DOUBLE: "f64",
    DataType.FLOAT: "f32",
    DataType.FLOAT16: "f16",
    DataType.FLOAT8E4M3FN: "f8e4m3fn",
    DataType.FLOAT8E5M2: "f8e5m2",
    DataType.FLOAT8E4M3FNUZ: "f8e4m3fnuz",
    DataType.FLOAT8E5M2FNUZ: "f8e5m2fnuz",
    DataType.FLOAT4E2M1: "f4e2m1",
    DataType.COMPLEX64: "c64",
    DataType.COMPLEX128: "c128",
    DataType.INT4: "i4",
    DataType.INT8: "i8",
    DataType.INT16: "i16",
    DataType.INT32: "i32",
    DataType.INT64: "i64",
    DataType.BOOL: "b8",
    DataType.UINT4: "u4",
    DataType.UINT8: "u8",
    DataType.UINT16: "u16",
    DataType.UINT32: "u32",
    DataType.UINT64: "u64",
    DataType.STRING: "s",
}

_SHORT_NAME_TO_DATA_TYPE = {v: k for k, v in _DATA_TYPE_TO_SHORT_NAME.items()}
