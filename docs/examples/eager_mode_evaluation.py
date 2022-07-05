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
from onnxscript import script
from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15 as op

@script()
def gemmgelu(A: FLOAT["N", "K"], W: FLOAT["K", "M"], Bias: FLOAT["M"]) -> FLOAT["N", "M"]:  # noqa F821

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


np.random.seed(0)
m = 2048
k = 16
n = 4096
a = np.random.rand(k, m).astype('float32').T
w = np.random.rand(n, k).astype('float32').T
b = np.random.rand(n,).astype('float32').T

print(gemmgelu(a, w, b))
