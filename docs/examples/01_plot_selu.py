"""
Generating a FunctionProto
==========================

The example below shows how we can define Selu as a function in onnxscript.
"""


# %%
# First, import the ONNX opset used to define the function.
from onnxscript import opset15 as op
from onnxscript import script

# %%
# Next, define Selu as an ONNXScript function.


@script()
def Selu(X, alpha: float, gamma: float):
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)


# %%
# We can convert the ONNXScript function to an ONNX function (FunctionProto) as below:

onnx_fun = Selu.to_function_proto()

# %%
# Let's see what the translated function looks like:
from onnxscript import proto2text  # noqa: E402

print(proto2text(onnx_fun))
