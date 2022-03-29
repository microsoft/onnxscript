"""
Define an onnx-script function/model.
=====================================
The examples below show how we can define ONNXScript functions and models.
"""

import onnxscript

# Import ONNX opset used in the function/model.
from onnxscript.onnx import opset15 as op

# Define an ONNXScript function.
@onnxscript.func
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

# Import tensor-type:
from onnxscript.onnx_types import FLOAT

# Define an ONNXScript model.
@onnxscript.model
def simple_one_layer_model(A: FLOAT[2048, 124], W: FLOAT[124, 4096],
         Bias: FLOAT[4096]) -> FLOAT[2048, 4096]:
    return op.Relu(op.MatMul(A, W) + Bias)

print('done!')