# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT
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
