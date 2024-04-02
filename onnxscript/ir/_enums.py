"""ONNX IR enums that matches the ONNX spec."""

from __future__ import annotations

import enum
import typing

if typing.TYPE_CHECKING:
    import numpy as np


class AttributeType(enum.IntEnum):
    """Enum for the types of ONNX attributes."""

    # TODO(justinchuby): Should we code gen this? We just need to get rid of protoc
    # We can code gen with https://github.com/recap-build/proto-schema-parser/tree/main
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

    @property
    def itemsize(self) -> float:
        """Returns the size of the data type in bytes."""
        return ITEMSIZE_MAP[self]

    def numpy(self) -> np.dtype:
        """Returns the numpy dtype for the ONNX data type.

        Raises:
            KeyError: If the data type is not supported by numpy.
        """
        import onnx.helper  # pylint: disable=import-outside-toplevel
        # Import here to avoid bringing in the onnx protobuf dependencies to the module

        return onnx.helper.tensor_dtype_to_np_dtype(self)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


ITEMSIZE_MAP = {
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
