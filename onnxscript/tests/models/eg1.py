# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT

# tensor inputs can have ONNX-like type annotations


def gemm(A: FLOAT[2048, 124], W: FLOAT[124, 4096], Bias: FLOAT[4096]) -> FLOAT[2048, 4096]:
    return op.MatMul(A, W) + Bias


# tensors and attributes distinguished by their types


def scale(A: FLOAT[...], alpha: float, beta: float) -> FLOAT[...]:
    return alpha * A + beta


# can return multiple-values


def prodsum(A: FLOAT["N"], B: FLOAT["N"]) -> (FLOAT["N"], FLOAT["N"]):
    prod = A * B
    sum = A + B
    return prod, sum


# can call ops/functions that return multiple-values


def dropout_eg(A: FLOAT[...]) -> FLOAT[...]:
    output, mask = op.Dropout(A, 0.7, True, seed=1729)
    return output


# will rename variable assigned multiple times


def renaming(A: FLOAT["N"]) -> FLOAT["N"]:
    T = op.Abs(A)
    T = op.Neg(T)
    return T
