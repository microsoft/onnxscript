"""
Generating a FunctionProto
=====================================

The example below shows how we can define Selu as a function in onnxscript.
"""


from onnxscript import script

#%%
# First, import the ONNX opset used to define the function.
from onnxscript.onnx_opset import opset15 as op

#%%
# Next, define Selu as an ONNXScript function.


@script()
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

#%%
# We can convert the ONNXScript function to an ONNX function (FunctionProto) as below:

onnx_fun = Selu.to_function_proto()

#%%
# Let's see what the translated function looks like:
import onnx
import onnx.printer
print(onnx.printer.to_text(onnx_fun))

#%%
# 