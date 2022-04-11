# SPDX-License-Identifier: Apache-2.0

from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx import opset15 as op


def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    if (sum1 < sum2):
        result = op.Identity(B)
    else:
        result = op.Identity(A)
    return result

# Test inference of inputs/outputs for then/else blocks:


def maxsum2(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    if (sum1 < sum2):
        temp = op.Identity(B)
        result = op.Identity(temp)
    else:
        temp = op.Identity(A)
        result = op.Identity(temp)
    return result

# test variables assigned only in one branch


def maxsum3(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    result = op.Identity(A)
    if (sum1 < sum2):
        result = op.Identity(B)
    return result


def check_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    if axis == zero:
        result = op.Transpose(x, perm=[1, 0])
    else:  # can we skip else?
        result = op.Identity(x)  # result = x does not work yet
    return result


def check_less_or_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    if axis <= zero:
        result = op.Transpose(x, perm=[1, 0])
    else:  # can we skip else?
        result = op.Identity(x)  # result = x does not work yet
    return result


def check_greater(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    if axis > zero:
        result = op.Transpose(x, perm=[1, 0])
    else:  # can we skip else?
        result = op.Identity(x)  # result = x does not work yet
    return result


def check_greater_or_equal(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    if axis >= zero:
        result = op.Transpose(x, perm=[1, 0])
    else:  # can we skip else?
        result = op.Identity(x)  # result = x does not work yet
    return result
