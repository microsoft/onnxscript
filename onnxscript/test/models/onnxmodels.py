# SPDX-License-Identifier: Apache-2.0

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
    return onnx.MatMul(A, W) + B


def gemmgelu(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:

    a = onnx.Constant(value_float=0.5)
    b = onnx.Constant(value_float=0.797885)
    c = onnx.Constant(value_float=0.035677)
    one = onnx.Constant(value_float=1.0)
    P1 = onnx.MatMul(A, W)
    X = onnx.Add(P1, Bias)
    T1 = onnx.Mul(X, X)
    T2 = onnx.Mul(c, T1)
    T3 = onnx.Add(b, T2)
    T4 = onnx.Mul(X, T3)
    T5 = onnx.Tanh(T4)
    T6 = onnx.Add(one, T5)
    T7 = onnx.Mul(X, T6)
    Y = onnx.Mul(a, T7)
    return Y


def gemmgelu2(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:

    a = onnx.Constant(value_float=0.5)
    b = onnx.Constant(value_float=0.797885)
    c = onnx.Constant(value_float=0.035677)
    one = onnx.Constant(value_float=1.0)
    P1 = onnx.MatMul(A, W)
    X = onnx.MatMul(A, W) + Bias
    T4 = X * (b + c * X * X)
    Y = a * X * (onnx.Tanh(T4) + one)
    return Y


def gemmgelu3(
        A: FLOAT[2048, 16],
        W: FLOAT[16, 4096],
        Bias: FLOAT[4096]
) -> FLOAT[2048, 4096]:
    a = 0.5
    b = 0.797885
    c = 0.035677
    P1 = onnx.MatMul(A, W)
    X = onnx.MatMul(A, W) + Bias
    T4 = X * (b + c * X * X)
    Y = a * X * (onnx.Tanh(T4) + 1.0)
    return Y
