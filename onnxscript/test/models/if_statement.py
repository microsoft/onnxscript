# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64


@script()
def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    if sum1 < sum2:
        result = op.Identity(B)
    else:
        result = op.Identity(A)
    return result


# Test inference of inputs/outputs for then/else blocks:


@script()
def maxsum2(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    if sum1 < sum2:
        temp = op.Identity(B)
        result = op.Identity(temp)
    else:
        temp = op.Identity(A)
        result = op.Identity(temp)
    return result


# test variables assigned only in one branch


@script()
def maxsum3(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    result = op.Identity(A)
    if sum1 < sum2:
        result = op.Identity(B)
    return result


@script()
def check_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if axis == zero:
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result


@script()
def check_less_or_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if axis <= zero:
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result


@script()
def check_greater(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if axis > zero:
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result


@script()
def check_greater_or_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if axis >= zero:
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result


@script()
def check_not(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if not (axis >= zero):
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result


@script()
def check_different(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if axis != zero:
        result = op.Transpose(x, perm=[1, 0])
    else:
        result = op.Identity(x)
    return result
