"""
Selu as a function
=====================================

The example below shows how we can define Selu as a function in onnxscript.
"""


from onnxscript import script

#%%
# Import ONNX opset used in the function/model.
from onnxscript.onnx import opset15 as op

#%%
# Define Selu as an ONNXScript function.


@script()
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

#%%
# Let's see how the translated function looks like.
import onnx
import onnx.printer
print(onnx.printer.to_text(Selu.to_function_proto()))

#%%
# 