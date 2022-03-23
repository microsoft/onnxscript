# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT


def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = oxs.ReduceSum(A)
    sum2 = oxs.ReduceSum(B)
    if (sum1 < sum2):
        result = oxs.Identity(B)
    else:
        result = oxs.Identity(A)
    return result

# Test inference of inputs/outputs for then/else blocks:


def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = oxs.ReduceSum(A)
    sum2 = oxs.ReduceSum(B)
    if (sum1 < sum2):
        temp = oxs.Identity(B)
        result = oxs.Identity(temp)
    else:
        temp = oxs.Identity(A)
        result = oxs.Identity(temp)
    return result

# test variables assigned only in one branch


def maxsum2(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = oxs.ReduceSum(A)
    sum2 = oxs.ReduceSum(B)
    result = oxs.Identity(A)
    if (sum1 < sum2):
        result = oxs.Identity(B)
    return result
