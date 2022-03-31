# SPDX-License-Identifier: Apache-2.0
from onnxscript import script
from onnxscript.onnx_types import FLOAT
from onnxscript.values import CustomOpset
from onnxscript.onnx import opset15 as op

opset = CustomOpset('this', 1)

@script(opset)
def MySelu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    neg = gamma * (alpha * op.Exp(X) - alpha)
    pos = gamma * X
    return op.Where(X <= 0, neg, pos)

@script(opset)
def MyElu(X: FLOAT[None], beta: FLOAT[1]) -> FLOAT[None]:
    alpha = op.Constant(value_float=1.)
    return MySelu(X, alpha, beta)
