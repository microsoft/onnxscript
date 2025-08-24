import numpy as np
import onnx
import onnx.helper as oh
import torch
from onnx import (
    TensorProto,
)


def np_dtype_to_tensor_dtype(dt: np.dtype) -> int:
    """Converts a numpy dtype into an ONNX element type.

    Args:
        dt: The numpy dtype to convert.

    Returns:
        The corresponding ONNX tensor type as an integer.

    Raises:
        ValueError: If the dtype cannot be converted to an ONNX type.
    """
    try:
        return oh.np_dtype_to_tensor_dtype(dt)
    except ValueError:
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None  # type: ignore
        if ml_dtypes is not None:
            if dt == ml_dtypes.bfloat16:
                return TensorProto.BFLOAT16
            if dt == ml_dtypes.float8_e4m3fn:
                return TensorProto.FLOAT8E4M3FN
            if dt == ml_dtypes.float8_e4m3fnuz:
                return TensorProto.FLOAT8E4M3FNUZ
            if dt == ml_dtypes.float8_e5m2:
                return TensorProto.FLOAT8E5M2
            if dt == ml_dtypes.float8_e5m2fnuz:
                return TensorProto.FLOAT8E5M2FNUZ
    if dt == np.float32:
        return TensorProto.FLOAT
    if dt == np.float16:
        return TensorProto.FLOAT16
    if dt == np.float64:
        return TensorProto.DOUBLE
    if dt == np.int64:
        return TensorProto.INT64
    if dt == np.uint64:
        return TensorProto.UINT64
    if dt == np.int16:
        return TensorProto.INT16
    if dt == np.uint16:
        return TensorProto.UINT16
    if dt == np.int32:
        return TensorProto.INT32
    if dt == np.int8:
        return TensorProto.INT8
    if dt == np.uint8:
        return TensorProto.UINT8
    if dt == np.uint32:
        return TensorProto.UINT32
    if dt == np.bool:
        return TensorProto.BOOL
    if dt == np.complex64:
        return TensorProto.COMPLEX64
    if dt == np.complex128:
        return TensorProto.COMPLEX128
    raise ValueError(f"Unable to convert type {dt}")


def torch_dtype_to_onnx_dtype(to: torch.dtype) -> int:
    """Converts a torch dtype into an ONNX element type.

    Args:
        to: The torch dtype to convert.

    Returns:
        The corresponding ONNX tensor type as an integer.

    Raises:
        NotImplementedError: If the torch dtype cannot be converted to an ONNX type.
    """
    if to == torch.float32:
        return onnx.TensorProto.FLOAT
    if to == torch.float16:
        return onnx.TensorProto.FLOAT16
    if to == torch.bfloat16:
        return onnx.TensorProto.BFLOAT16
    if to == torch.float64:
        return onnx.TensorProto.DOUBLE
    if to == torch.int64:
        return onnx.TensorProto.INT64
    if to == torch.int32:
        return onnx.TensorProto.INT32
    if to == torch.uint64:
        return onnx.TensorProto.UINT64
    if to == torch.uint32:
        return onnx.TensorProto.UINT32
    if to == torch.bool:
        return onnx.TensorProto.BOOL
    if to == torch.SymInt:
        return onnx.TensorProto.INT64
    if to == torch.int16:
        return onnx.TensorProto.INT16
    if to == torch.uint16:
        return onnx.TensorProto.UINT16
    if to == torch.int8:
        return onnx.TensorProto.INT8
    if to == torch.uint8:
        return onnx.TensorProto.UINT8
    if to == torch.SymFloat:
        return onnx.TensorProto.FLOAT
    if to == torch.complex64:
        return onnx.TensorProto.COMPLEX64
    if to == torch.complex128:
        return onnx.TensorProto.COMPLEX128
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")
