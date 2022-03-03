# SPDX-License-Identifier: Apache-2.0

from onnxscript._types import FLOAT


def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = onnx.ReduceSum(A)
    sum2 = onnx.ReduceSum(B)
    if (sum1 < sum2):
        result = onnx.Identity(B)
    else:
        result = onnx.Identity(A)
    return result

# Test inference of inputs/outputs for then/else blocks:


def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = onnx.ReduceSum(A)
    sum2 = onnx.ReduceSum(B)
    if (sum1 < sum2):
        temp = onnx.Identity(B)
        result = onnx.Identity(temp)
    else:
        temp = onnx.Identity(A)
        result = onnx.Identity(temp)
    return result

# test variables assigned only in one branch


def maxsum2(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = onnx.ReduceSum(A)
    sum2 = onnx.ReduceSum(B)
    result = onnx.Identity(A)
    if (sum1 < sum2):
        result = onnx.Identity(B)
    return result
