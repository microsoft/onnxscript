# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import abc
from typing import ClassVar, Optional, Tuple, Union

import onnx
import onnx.helper

import onnxscript.ir

_DType = onnxscript.ir.DataType
_DimType = Union[int, str, type(None)]
_ShapeType = Union[Tuple[_DimType, ...], _DimType, type(Ellipsis)]

_tensor_type_shape_cache: dict[_DType, TensorType] = {}
tensor_type_registry: dict[_DType, TensorType] = {}


def _check_dim(dim):
    if not isinstance(dim, (int, str, type(None))):
        raise TypeError(f"Invalid dimension {dim}")


def _check_shape(shape):
    if isinstance(shape, tuple):
        for dim in shape:
            _check_dim(dim)
    elif shape != Ellipsis:
        _check_dim(shape)


class TensorType(abc.ABC):
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

    dtype: ClassVar[_DType]
    shape: ClassVar[Optional[_ShapeType]]

    def __new__(cls):
        raise NotImplementedError("TensorTypes cannot be instantiated")

    def __init_subclass__(cls, dtype: _DType, shape: Optional[_ShapeType] = None):
        cls.dtype = dtype
        cls.shape = shape
        if shape is None:
            existing_cls = tensor_type_registry.get(dtype)
            if existing_cls is not None:
                raise ValueError(
                    f"Invalid usage: subclass {existing_cls!r} "
                    f"already defined for dtype={dtype}"
                )
            tensor_type_registry[dtype] = cls
        else:
            _check_shape(shape)

    def __class_getitem__(cls, shape: Optional[_ShapeType]) -> type[TensorType]:
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

    @classmethod
    def to_string(cls) -> str:
        return f"tensor({cls.__name__.lower()})"


class FLOAT(TensorType, dtype=onnxscript.ir.DataType.FLOAT):
    pass


class UINT8(TensorType, dtype=onnxscript.ir.DataType.UINT8):
    pass


class INT8(TensorType, dtype=onnxscript.ir.DataType.INT8):
    pass


class UINT16(TensorType, dtype=onnxscript.ir.DataType.UINT16):
    pass


class INT16(TensorType, dtype=onnxscript.ir.DataType.INT16):
    pass


class INT32(TensorType, dtype=onnxscript.ir.DataType.INT32):
    pass


class INT64(TensorType, dtype=onnxscript.ir.DataType.INT64):
    pass


class STRING(TensorType, dtype=onnxscript.ir.DataType.STRING):
    pass


class BOOL(TensorType, dtype=onnxscript.ir.DataType.BOOL):
    pass


class FLOAT16(TensorType, dtype=onnxscript.ir.DataType.FLOAT16):
    pass


class DOUBLE(TensorType, dtype=onnxscript.ir.DataType.DOUBLE):
    pass


class UINT32(TensorType, dtype=onnxscript.ir.DataType.UINT32):
    pass


class UINT64(TensorType, dtype=onnxscript.ir.DataType.UINT64):
    pass


class COMPLEX64(TensorType, dtype=onnxscript.ir.DataType.COMPLEX64):
    pass


class COMPLEX128(TensorType, dtype=onnxscript.ir.DataType.COMPLEX128):
    pass


class BFLOAT16(TensorType, dtype=onnxscript.ir.DataType.BFLOAT16):
    pass


class FLOAT8E4M3FN(TensorType, dtype=onnxscript.ir.DataType.FLOAT8E4M3FN):
    pass


class FLOAT8E4M3FNUZ(TensorType, dtype=onnxscript.ir.DataType.FLOAT8E4M3FNUZ):
    pass


class FLOAT8E5M2(TensorType, dtype=onnxscript.ir.DataType.FLOAT8E5M2):
    pass


class FLOAT8E5M2FNUZ(TensorType, dtype=onnxscript.ir.DataType.FLOAT8E5M2FNUZ):
    pass


class INT4(TensorType, dtype=onnxscript.ir.DataType.INT4):
    pass


class UINT4(TensorType, dtype=onnxscript.ir.DataType.UINT4):
    pass


class FLOAT4E2M1(TensorType, dtype=onnxscript.ir.DataType.FLOAT4E2M1):
    pass


def onnx_type_to_onnxscript_repr(onnx_type: onnx.TypeProto) -> str:
    """Converts an onnx type into the string representation of the type in *onnxscript*.

    Args:
        onnx_type: an instance of onnx TypeProto

    Returns:
        The string representation of the type in onnxscript

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
            if not shape:
                return name
            return f"{name}[{','.join(shape)}]"
        return f"{name}[...]"
    raise NotImplementedError(f"Unable to translate type {onnx_type!r} into onnxscript type.")


# Currently, only tensor types are supported. Need to expand support for other ONNX types.
ONNXType = TensorType
