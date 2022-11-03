# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

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

    default_instance: Optional["TensorType"] = None

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

    def __class_getitem__(cls, shape: Optional[ShapeType]):
        if cls.default_instance is None:
            raise TypeError(f"{cls} does not specify a default_instance.")
        # pylint erroneously flags with unsubscriptable-object if
        # using subscript notation (cls.default_instance[shape]):
        return cls.default_instance.__getitem__(shape)

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


class _BuiltinTensorType:
    def __init__(self, tensor_proto: onnx.TensorProto):
        self.tensor_proto = tensor_proto

    def __call__(self, cls):
        cls.default_instance = TensorType(self.tensor_proto)
        cls.to_type_proto = cls.default_instance.to_type_proto
        return cls


@_BuiltinTensorType(onnx.TensorProto.FLOAT)
class FLOAT(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.UINT8)
class UINT8(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.INT8)
class INT8(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.UINT16)
class UINT16(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.INT16)
class INT16(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.INT32)
class INT32(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.INT64)
class INT64(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.STRING)
class STRING(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.BOOL)
class BOOL(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.FLOAT16)
class FLOAT16(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.DOUBLE)
class DOUBLE(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.UINT32)
class UINT32(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.UINT64)
class UINT64(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.COMPLEX64)
class COMPLEX64(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.COMPLEX128)
class COMPLEX128(TensorType):
    pass


@_BuiltinTensorType(onnx.TensorProto.BFLOAT16)
class BFLOAT16(TensorType):
    pass


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


# Currently, only tensor types are supported. Need to expand support for other ONNX types.
ONNXType = TensorType
