# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def gemmgelu(
    A: FLOAT["M", "K"],  # noqa: F821
    W: FLOAT["K", "N"],  # noqa: F821
    Bias: FLOAT["N"],  # noqa: F821
) -> FLOAT["M", "N"]:  # noqa: F821
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
