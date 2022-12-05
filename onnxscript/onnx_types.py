# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import ClassVar, Optional, Tuple, Union

import onnx
import onnx.helper

DType = onnx.TensorProto.DataType

DimType = Union[int, str, type(None)]


def _check_dim(dim):
    if not isinstance(dim, (int, str, type(None))):
        raise TypeError(f"Invalid dimension {dim}")


ShapeType = Union[Tuple[DimType, ...], DimType, type(Ellipsis)]


def _check_shape(shape):
    if isinstance(shape, tuple):
        for dim in shape:
            _check_dim(dim)
    elif shape != Ellipsis:
        _check_dim(shape)


_tensor_type_registry: dict[DType, TensorType] = {}
_tensor_type_shape_cache: dict[DType, TensorType] = {}


class TensorType(type):
    """ONNX Script representation of a tensor type supporting shape annotations.

    A scalar-tensor of rank 0:
    ::

        tensor: FLOAT

    A tensor of unknown rank:
    ::

        tensor: FLOAT[...]

    A tensor of rank 2 of unknown dimensions, with symbolic names:
    ::

        tensor: FLOAT['M', 'N']

    A tensor of rank 2 of known dimensions:
    ::

        tensor: FLOAT[128, 1024]
    """

    dtype: ClassVar[DType]
    shape: ClassVar[Optional[ShapeType]]

    def __new__(cls):
        raise NotImplementedError("TensorTypes cannot be instantiated")

    def __init__(cls):
        raise NotImplementedError("TensorTypes cannot be instantiated")

    def __init_subclass__(cls, dtype: DType, shape: Optional[ShapeType] = None):
        cls.dtype = dtype
        cls.shape = shape
        if shape is None:
            existing_cls = _tensor_type_registry.get(dtype)
            if existing_cls is not None:
                raise ValueError(
                    f"Invalid usage: subclass {existing_cls!r} "
                    f"already defined for dtype={dtype}"
                )
            _tensor_type_registry[dtype] = cls
        else:
            _check_shape(shape)

    def __getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        if cls.shape is not None:
            raise ValueError("Invalid usage: shape already specified.")
        if shape is None:
            # Treat FLOAT[NONE] as 1-dimensional tensor with unknown dimension
            shape = (None,)
        key = (cls.dtype, shape)
        shaped_type = _tensor_type_shape_cache.get(key)
        if shaped_type is None:
            shaped_type = type(cls.__name__, (TensorType,), {}, dtype=cls.dtype, shape=shape)
            _tensor_type_shape_cache[key] = shaped_type
        return shaped_type

    @classmethod
    def to_type_proto(cls) -> onnx.TypeProto:
        if cls.shape is None:
            shape = ()  # "FLOAT" is treated as a scalar
        elif cls.shape is Ellipsis:
            shape = None  # "FLOAT[...]" is a tensor of unknown rank
        elif isinstance(cls.shape, tuple):
            shape = cls.shape  # example: "FLOAT[10,20]"
        else:
            shape = [cls.shape]  # example: "FLOAT[10]"
        return onnx.helper.make_tensor_type_proto(cls.dtype, shape)


# pylint: disable=abstract-method,too-many-function-args
class FLOAT(TensorType, dtype=onnx.TensorProto.FLOAT):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class UINT8(TensorType, dtype=onnx.TensorProto.UINT8):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class INT8(TensorType, dtype=onnx.TensorProto.INT8):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class UINT16(TensorType, dtype=onnx.TensorProto.UINT16):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class INT16(TensorType, dtype=onnx.TensorProto.INT16):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class INT32(TensorType, dtype=onnx.TensorProto.INT32):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class INT64(TensorType, dtype=onnx.TensorProto.INT64):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class STRING(TensorType, dtype=onnx.TensorProto.STRING):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class BOOL(TensorType, dtype=onnx.TensorProto.BOOL):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class FLOAT16(TensorType, dtype=onnx.TensorProto.FLOAT16):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class DOUBLE(TensorType, dtype=onnx.TensorProto.DOUBLE):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class UINT32(TensorType, dtype=onnx.TensorProto.UINT32):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class UINT64(TensorType, dtype=onnx.TensorProto.UINT64):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class COMPLEX64(TensorType, dtype=onnx.TensorProto.COMPLEX64):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class COMPLEX128(TensorType, dtype=onnx.TensorProto.COMPLEX128):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


class BFLOAT16(TensorType, dtype=onnx.TensorProto.BFLOAT16):
    def __class_getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        return super().__getitem__(cls, shape)


# pylint: enable=abstract-method,too-many-function-args


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
