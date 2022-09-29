# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Union
import onnx
import onnx.helper


class TensorType:
    # Reference implementation placeholder
    # represents a generic ONNX tensor type
    def __init__(self, dtype=onnx.TensorProto.UNDEFINED, shape=None) -> None:
        self.dtype = dtype
        self.shape = shape

    def __str__(self) -> str:
        shapestr = str(self.shape) if self.shape else "[...]"
        return onnx.TensorProto.DataType.Name(self.dtype) + shapestr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype!r}, shape={self.shape!r})"

    def to_type_proto(self):
        # TODO: handle None
        return onnx.helper.make_tensor_type_proto(self.dtype, self.shape)


# Utilities used to create parametrized type-annotations for tensors.
# Example type annotations:
#    x : FLOAT (a scalar-tensor of rank 0)
#    x : FLOAT[...] (a tensor of unknown rank)
#    x : FLOAT['M', 'N'] (a tensor of rank 2 of unknown dimensions, with symbolic names)
#    x : FLOAT[128, 1024] (a tensor of rank 2 of known dimensions)


class ParametricTensor:
    """
    Defines a dense tensor of any shape.
    """

    types = {}

    def __init__(self, dtype) -> None:
        self.dtype = dtype
        ParametricTensor.types[dtype] = self

    def __getitem__(self, shape):
        def mk_dim(dim):  # pylint: disable=unused-variable # TODO: why?
            r = onnx.TensorShapeProto.Dimension()
            if isinstance(dim, int):
                r.dim_value = dim
            elif isinstance(dim, str):
                r.dim_param = dim
            elif dim is not None:
                raise TypeError("Invalid dimension")
            return r

        if isinstance(shape, tuple):
            s = shape
        elif shape == Ellipsis:
            s = None
        else:
            s = [shape]
        return TensorType(self.dtype, s)

    def to_type_proto(self):
        return onnx.helper.make_tensor_type_proto(self.dtype, ())

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

# TODO: add type annotations for the other ONNX types (Optional, Sequence, Map)
ONNXType = Union[TensorType, ParametricTensor]
