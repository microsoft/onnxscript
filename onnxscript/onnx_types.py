# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Optional, Tuple, Union

import onnx
import onnx.helper

# Representations of ONNX types in ONNX Script.
# Currently restricted to tensor types.
# Example type annotations in ONNX Script.
#    x : FLOAT (a scalar-tensor of rank 0)
#    x : FLOAT[...] (a tensor of unknown rank)
#    x : FLOAT['M', 'N'] (a tensor of rank 2 of unknown dimensions, with symbolic names)
#    x : FLOAT[128, 1024] (a tensor of rank 2 of known dimensions)

DimType = Union[int, str, type(None)]


def check_dim(dim):
    if not isinstance(dim, (int, str, type(None))):
        raise TypeError(f"Invalid dimension {dim}")


ShapeType = Union[Tuple[DimType, ...], DimType, type(Ellipsis)]


def check_shape(shape):
    if isinstance(shape, tuple):
        for dim in shape:
            check_dim(dim)
    elif shape != Ellipsis:
        check_dim(shape)


class TensorType:
    """ONNX Script representation of a tensor type."""

    def __init__(self, dtype, shape: Optional[ShapeType] = None) -> None:
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            check_shape(shape)

    def __getitem__(self, shape: Optional[ShapeType]):
        if self.shape is not None:
            raise ValueError("Invalid usage: shape already specified.")
        if shape is None:
            # Treat FLOAT[NONE] as 1-dimensional tensor with unknown dimension
            shape = (None,)
        return TensorType(self.dtype, shape)

    def to_type_proto(self) -> onnx.TypeProto:
        if self.shape is None:
            shape = ()  # "FLOAT" is treated as a scalar
        elif self.shape is Ellipsis:
            shape = None  # "FLOAT[...]" is a tensor of unknown rank
        elif isinstance(self.shape, tuple):
            shape = self.shape  # example: "FLOAT[10,20]"
        else:
            shape = [self.shape]  # example: "FLOAT[10]"
        return onnx.helper.make_tensor_type_proto(self.dtype, shape)


FLOAT = TensorType(onnx.TensorProto.FLOAT)
UINT8 = TensorType(onnx.TensorProto.UINT8)
INT8 = TensorType(onnx.TensorProto.INT8)
UINT16 = TensorType(onnx.TensorProto.UINT16)
INT16 = TensorType(onnx.TensorProto.INT16)
INT32 = TensorType(onnx.TensorProto.INT32)
INT64 = TensorType(onnx.TensorProto.INT64)
STRING = TensorType(onnx.TensorProto.STRING)
BOOL = TensorType(onnx.TensorProto.BOOL)
FLOAT16 = TensorType(onnx.TensorProto.FLOAT16)
DOUBLE = TensorType(onnx.TensorProto.DOUBLE)
UINT32 = TensorType(onnx.TensorProto.UINT32)
UINT64 = TensorType(onnx.TensorProto.UINT64)
COMPLEX64 = TensorType(onnx.TensorProto.COMPLEX64)
COMPLEX128 = TensorType(onnx.TensorProto.COMPLEX128)
BFLOAT16 = TensorType(onnx.TensorProto.BFLOAT16)


def onnx_type_to_onnxscript_repr(onnx_type: onnx.TypeProto) -> str:
    """Converts an onnx type into the string representation of the type in *onnx-script*.
    Args:
        onnx_type: an instance of onnx TypeProto

    Returns:
        The string representation of the type in onnx-script

    Raises:
        ...
    """
    if onnx_type.HasField("tensor_type"):
        elem_type = onnx_type.tensor_type.elem_type
        name = onnx.TensorProto.DataType.Name(elem_type)
        if onnx_type.tensor_type.HasField("shape"):
            shape = []
            for d in onnx_type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(str(d.dim_value))
                elif d.HasField("dim_param"):
                    shape.append(repr(d.dim_param))
                else:
                    shape.append("None")
            if len(shape) == 0:
                return name
            return f"{name}[{','.join(shape)}]"
        return f"{name}[...]"
    raise NotImplementedError(f"Unable to translate type {onnx_type!r} into onnx-script type.")
