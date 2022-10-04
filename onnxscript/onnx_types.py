# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import onnx.helper
from typing import Union, Tuple, Optional


# Utilities used to create parametrized type-annotations for tensors.
# Example type annotations:
#    x : FLOAT (a scalar-tensor of rank 0)
#    x : FLOAT[...] (a tensor of unknown rank)
#    x : FLOAT['M', 'N'] (a tensor of rank 2 of unknown dimensions, with symbolic names)
#    x : FLOAT[128, 1024] (a tensor of rank 2 of known dimensions)

DimType = Union[int, str, type(None)]

def check_dim(dim):
    if not isinstance(dim, (int, str, type(None))):
        raise TypeError(f"Invalid dimension {dim}")

ShapeType = Union[Tuple[DimType], DimType, type(Ellipsis)]

def check_shape(shape):
    if isinstance(shape, tuple):
        for dim in shape:
            check_dim(dim)
    elif shape != Ellipsis:
        check_dim(shape)

class ParametricTensor:
    """Representation of a tensor type."""

    def __init__(self, dtype, shape : Optional[ShapeType] = None) -> None:
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            check_shape(shape)

    def __getitem__(self, shape: Optional[ShapeType]):
        if shape is None:
            # Treat FLOAT[NONE] as 1-dimensional tensor with unknown dimension
            shape = (None,)
        return ParametricTensor(self.dtype, shape)

    def to_type_proto(self):
        if self.shape is None:
            shape = ()
        elif self.shape is Ellipsis:
            shape = None
        elif isinstance(self.shape, tuple):
            shape = self.shape
        else:
            shape = [self.shape]
        return onnx.helper.make_tensor_type_proto(self.dtype, shape)

    def __repr__(self) -> str:
        return onnx.TensorProto.DataType.Name(self.dtype)

FLOAT = ParametricTensor(onnx.TensorProto.FLOAT)
UINT8 = ParametricTensor(onnx.TensorProto.UINT8)
INT8 = ParametricTensor(onnx.TensorProto.INT8)
UINT16 = ParametricTensor(onnx.TensorProto.UINT16)
INT16 = ParametricTensor(onnx.TensorProto.INT16)
INT32 = ParametricTensor(onnx.TensorProto.INT32)
INT64 = ParametricTensor(onnx.TensorProto.INT64)
STRING = ParametricTensor(onnx.TensorProto.STRING)
BOOL = ParametricTensor(onnx.TensorProto.BOOL)
FLOAT16 = ParametricTensor(onnx.TensorProto.FLOAT16)
DOUBLE = ParametricTensor(onnx.TensorProto.DOUBLE)
UINT32 = ParametricTensor(onnx.TensorProto.UINT32)
UINT64 = ParametricTensor(onnx.TensorProto.UINT64)
COMPLEX64 = ParametricTensor(onnx.TensorProto.COMPLEX64)
COMPLEX128 = ParametricTensor(onnx.TensorProto.COMPLEX128)
BFLOAT16 = ParametricTensor(onnx.TensorProto.BFLOAT16)

def get_repr(onnx_dtype):
    return onnx.TensorProto.DataType.Name(onnx_dtype)
