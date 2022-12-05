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


_tensor_type_shape_cache: dict[DType, TensorType] = {}


class _WithOnnxType:
    """Class that implements to_type_proto."""

    dtype: ClassVar[DType]
    shape: ClassVar[Optional[ShapeType]] = None

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
    shape: ClassVar[Optional[ShapeType]] = None

    def __getitem__(cls, shape: Optional[ShapeType]) -> type[TensorType]:
        if cls.shape is not None:
            raise ValueError("Invalid usage: shape already specified.")
        if shape is None:
            # Treat FLOAT[NONE] as 1-dimensional tensor with unknown dimension
            shape = (None,)
        _check_shape(shape)
        key = (cls.dtype, shape)
        shaped_type = _tensor_type_shape_cache.get(key)
        if shaped_type is None:
            # This calls __init_subclass__
            shaped_type = type(
                cls.__name__,
                (type(cls),),
                dict(dtype=cls.dtype, shape=shape),
            )
            _tensor_type_shape_cache[key] = shaped_type
        return shaped_type


# pylint: disable=abstract-method,too-many-function-args
class FLOAT(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.FLOAT
    shape = None


class UINT8(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.UINT8
    shape = None


class INT8(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.INT8
    shape = None


class UINT16(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.UINT16
    shape = None


class INT16(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.INT16
    shape = None


class INT32(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.INT32
    shape = None


class INT64(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.INT64
    shape = None


class STRING(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.STRING
    shape = None


class BOOL(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.BOOL
    shape = None


class FLOAT16(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.FLOAT16
    shape = None


class DOUBLE(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.DOUBLE
    shape = None


class UINT32(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.UINT32
    shape = None


class UINT64(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.UINT64
    shape = None


class COMPLEX64(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.COMPLEX64
    shape = None


class COMPLEX128(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.COMPLEX128
    shape = None


class BFLOAT16(_WithOnnxType, metaclass=TensorType):
    dtype = onnx.TensorProto.BFLOAT16
    shape = None


# pylint: enable=abstract-method,too-many-function-args

_tensor_type_registry: dict[DType, TensorType] = {
    onnx.TensorProto.FLOAT: FLOAT,
    onnx.TensorProto.UINT8: UINT8,
    onnx.TensorProto.INT8: INT8,
    onnx.TensorProto.UINT16: UINT16,
    onnx.TensorProto.INT16: INT16,
    onnx.TensorProto.INT32: INT32,
    onnx.TensorProto.INT64: INT64,
    onnx.TensorProto.STRING: STRING,
    onnx.TensorProto.BOOL: BOOL,
    onnx.TensorProto.FLOAT16: FLOAT16,
    onnx.TensorProto.DOUBLE: DOUBLE,
    onnx.TensorProto.UINT32: UINT32,
    onnx.TensorProto.UINT64: UINT64,
    onnx.TensorProto.COMPLEX64: COMPLEX64,
    onnx.TensorProto.COMPLEX128: COMPLEX128,
    onnx.TensorProto.BFLOAT16: BFLOAT16,
}


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
