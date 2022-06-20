# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16


@script(Opset('this', 1))
def SeluSubFunction15(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op15.Constant(value_float=1.)
    neg = gamma * (alpha * op15.Exp(X) - alpha)
    pos = gamma * X
    return op15.Where(X <= zero, neg, pos)

@script(Opset('this', 1))
def SeluSubFunction16(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op16.Constant(value_float=1.)
    neg = gamma * (alpha * op16.Exp(X) - alpha)
    pos = gamma * X
    return op16.Where(X <= zero, neg, pos)

# onnx-script function, version 1, using onnx opset15
@script(Opset('this', 1))
def Elu15(X: FLOAT[None], beta: FLOAT[1]=1.0) -> FLOAT[None]:
    alpha = op15.Constant(value_float=1.)
    return SeluSubFunction15(X, alpha, beta)

# onnx-script function, version 2, using onnx opset16
@script(Opset('this', 2))
def Elu16(X: FLOAT[None], beta: FLOAT[1]=1.0) -> FLOAT[None]:
    alpha = op16.Constant(value_float=1.)
    return SeluSubFunction16(X, alpha, beta)

import math
@script(Opset('this', 1))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    half = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=1/math.sqrt(2))
    one = op15.Constant(value_float=1.0)
    P1 = op15.MatMul(A, W)
    X = op15.Add(P1, Bias)
    T1 = op15.erf(X, b)
    T2 = op15.Add(one, T1)
    T3 = op15.Mul(half, T2)
    Y = op15.Mul(X, T3)
    return Y

@script(Opset('this', 1))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    half = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=1.0/math.sqrt(2.0))
    one = op15.Constant(value_float=1.0)

    X = op15.Gemm(A, W, Bias)

    T1 = op15.erf(X, b)
    T2 = op15.Add(one, T1)
    T3 = op15.Mul(half, T2)
    Y = op15.Mul(X, T3)
    return Y

@script(Opset('this', 2))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    a = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=0.797885)
    c = op15.Constant(value_float=0.035677)
    one = op15.Constant(value_float=1.0)

    X = op15.Gemm(A, W, Bias)

    T1 = op15.Mul(X, X)
    T2 = op15.Mul(c, T1)
    T3 = op15.Add(b, T2)
    T4 = op15.Mul(X, T3)
    T5 = op15.Tanh(T4)
    T6 = op15.Add(one, T5)
    T7 = op15.Mul(X, T6)
    Y = op15.Mul(a, T7)
    return Y
