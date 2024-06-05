# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common operators shared in the torchlib library."""

import onnxscript
import onnxscript.values
from onnxscript import BOOL, INT64
from onnxscript import opset18 as op
from onnxscript.function_libs.torch_lib import _constants, tensor_typing
from onnxscript.function_libs.torch_lib.tensor_typing import RealType
from onnxscript.onnx_types import COMPLEX64, COMPLEX128, DOUBLE, FLOAT

COMPLEX64_TYPE = COMPLEX64.dtype
COMPLEX128_TYPE = COMPLEX128.dtype

DOMAIN = f"{_constants.DOMAIN}.common"

common_opset = onnxscript.values.Opset(domain=DOMAIN, version=1)


@onnxscript.script(common_opset)
def Rank(input: tensor_typing.TTensor) -> INT64:
    """Take the rank of the input tensor."""

    return op.Size(op.Shape(input))


@onnxscript.script(common_opset)
def IsScalar(input: tensor_typing.TTensor) -> BOOL:
    """Return whether the input has rank 0, or is a scalar."""

    return op.Equal(op.Size(op.Shape(input)), op.Constant(value_int=0))


def cast_to(a: RealType, dtype: int) -> RealType:
    """Cast input to dtype while handling complex types."""

    # Traced function because different if branches return different dtypes
    # which is not supported in an ONNX function
    if dtype == COMPLEX128_TYPE:
        # Cast to the real representation of the complex type
        casted = op.Cast(a, to=DOUBLE.dtype)
        # Create a complex number
        real_part = op.Unsqueeze(casted, axes=[-1])
        imag_part = op.Expand(op.Cast(0.0, to=DOUBLE.dtype), op.Shape(real_part))
        result = op.Concat(real_part, imag_part, axis=-1)
    elif dtype == COMPLEX64_TYPE:
        # Cast to the real representation of the complex type
        casted = op.Cast(a, to=FLOAT.dtype)
        # Create a complex number
        real_part = op.Unsqueeze(casted, axes=[-1])
        imag_part = op.Expand(0.0, op.Shape(real_part))
        result = op.Concat(real_part, imag_part, axis=-1)
    else:
        # Cast to real numbers
        result = op.Cast(a, to=dtype)

    return result
