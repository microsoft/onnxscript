# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import CustomOpset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16


@script(CustomOpset('this', 1))
def SeluSubFunction15(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op15.Constant(value_float=1.)
    neg = gamma * (alpha * op15.Exp(X) - alpha)
    pos = gamma * X
    return op15.Where(X <= zero, neg, pos)

@script(CustomOpset('this', 1))
def SeluSubFunction16(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op16.Constant(value_float=1.)
    neg = gamma * (alpha * op16.Exp(X) - alpha)
    pos = gamma * X
    return op16.Where(X <= zero, neg, pos)

# onnx-script function, version 1, using onnx opset15
@script(CustomOpset('this', 1))
def Elu15(X: FLOAT[None], beta: FLOAT[1]=1.0) -> FLOAT[None]:
    alpha = op15.Constant(value_float=1.)
    return SeluSubFunction15(X, alpha, beta)

# onnx-script function, version 2, using onnx opset16
@script(CustomOpset('this', 2))
def Elu16(X: FLOAT[None], beta: FLOAT[1]=1.0) -> FLOAT[None]:
    alpha = op16.Constant(value_float=1.)
    return SeluSubFunction16(X, alpha, beta)
