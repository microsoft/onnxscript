# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_types import FLOAT
from onnxscript.onnx import opset15 as op


@script()
def MySelu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    neg = gamma * (alpha * op.Exp(X) - alpha)
    pos = gamma * X
    return op.Where(X <= 0, neg, pos)

@script()
def MyElu(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.)
    return MySelu(X, alpha, beta)

@script()
def MyEluB(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.)
    res = MySelu(X, alpha, beta)
    return res

@script()
def MyEluC(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.)
    res = op.Identity(MySelu(X, alpha, beta))
    return res
