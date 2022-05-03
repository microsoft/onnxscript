# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT

# type annotation tests:


def ta_test1(A: FLOAT[100], B: FLOAT[100]) -> FLOAT[100]:
    return A + B


def ta_test2(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    return A + B

# feature: constant promotion


def plus1(A: FLOAT["N"]) -> FLOAT["N"]:
    return A + 1.0

# example models


def eg1(A: FLOAT[100, 200], W: FLOAT[200, 96], B: FLOAT[96]) -> FLOAT[100, 96]:
    return op.MatMul(A, W) + B


def gemmgelu(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:

    a = op.Constant(value_float=0.5)
    b = op.Constant(value_float=0.797885)
    c = op.Constant(value_float=0.035677)
    one = op.Constant(value_float=1.0)
    P1 = op.MatMul(A, W)
    X = op.Add(P1, Bias)
    T1 = op.Mul(X, X)
    T2 = op.Mul(c, T1)
    T3 = op.Add(b, T2)
    T4 = op.Mul(X, T3)
    T5 = op.Tanh(T4)
    T6 = op.Add(one, T5)
    T7 = op.Mul(X, T6)
    Y = op.Mul(a, T7)
    return Y


def gemmgelu2(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:

    a = op.Constant(value_float=0.5)
    b = op.Constant(value_float=0.797885)
    c = op.Constant(value_float=0.035677)
    one = op.Constant(value_float=1.0)
    P1 = op.MatMul(A, W)
    X = op.MatMul(A, W) + Bias
    T4 = X * (b + c * X * X)
    Y = a * X * (op.Tanh(T4) + one)
    return Y


def gemmgelu3(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:
    a = 0.5
    b = 0.797885
    c = 0.035677
    P1 = op.MatMul(A, W)
    X = op.MatMul(A, W) + Bias
    T4 = X * (b + c * X * X)
    Y = a * X * (op.Tanh(T4) + 1.0)
    return Y
