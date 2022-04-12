"""
Define an onnx-script function/model.
=====================================
The examples below show how we can define ONNXScript functions and models.
"""


from onnxscript import script

# Import ONNX opset used in the function/model.
from onnxscript.onnx import opset15 as op

# Import tensor-type:
from onnxscript.onnx_types import FLOAT

# Define an ONNXScript function.


@script()
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)
