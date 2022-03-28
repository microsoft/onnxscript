"""
Debug an onnx-script function in eager mode.
=================================================

This example demonstrates a way to debug an *onnx-script* function
with eager_mode_evaluator. Users can step into
the gemmgelu function and step over each onnx op call.
Intermediate variables are available for troubleshooting.
eager_mode_evaluator uses onnxruntime
as the backend to compute onnx ops.
"""
import numpy as np
from onnxscript import eager_mode_evaluator as oxs
from onnxscript.onnx_types import FLOAT


def gemmgelu(A: FLOAT["N", "K"], W: FLOAT["K", "M"], Bias: FLOAT["M"] ) -> FLOAT["N", "M"]: # noqa F821

    a = oxs.Constant(value_float=0.5)
    b = oxs.Constant(value_float=0.797885)
    c = oxs.Constant(value_float=0.035677)
    one = oxs.Constant(value_float=1.0)
    P1 = oxs.MatMul(A, W)
    X = oxs.Add(P1, Bias)
    print("X: ", X)
    T1 = oxs.Mul(X, X)
    T2 = oxs.Mul(c, T1)
    T3 = oxs.Add(b, T2)
    T4 = oxs.Mul(X, T3)
    T5 = oxs.Tanh(T4)
    T6 = oxs.Add(one, T5)
    T7 = oxs.Mul(X, T6)
    Y = oxs.Mul(a, T7)
    return Y


np.random.seed(0)
m = 2048
k = 16
n = 4096
a = np.random.rand(k, m).astype('float32').T
w = np.random.rand(n, k).astype('float32').T
b = np.random.rand(n,).astype('float32').T

print(gemmgelu(a, w, b))
