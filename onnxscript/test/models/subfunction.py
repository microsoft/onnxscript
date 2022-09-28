# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def MySelu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op.Constant(value_float=1.0)
    neg = gamma * (alpha * op.Exp(X) - alpha)
    pos = gamma * X
    return op.Where(X <= zero, neg, pos)


@script()
def MyElu(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.0)
    return MySelu(X, alpha, beta)


@script()
def MyEluB(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.0)
    res = MySelu(X, alpha, beta)
    return res


@script()
def MyEluC(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.0)
    res = op.Identity(MySelu(X, alpha, beta))
    return res


@script()
def MyEluD(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    res = op.Identity(MyEluC(X, beta))
    return res


@script()
def IfMyEluD(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    zero = op.Constant(value_float=1.0)
    if beta > zero:
        result = MyEluB(X, beta)
    else:
        result = MyEluC(X, beta)
    return result
