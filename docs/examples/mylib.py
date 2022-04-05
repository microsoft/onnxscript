"""
Define an onnx-script library consisting of multiple functions.
==============================================================
The examples below show how we can define a library consisting
of multiple functions.

See exportlib.py for how to export an ONNX library proto file.
"""

from onnxscript import script
from onnxscript.onnx import opset15 as op

# The domain/version of the library functions defined below
opset = CustomOpset('com.mydomain', 1)


@script(opset)
def l2norm(X):
    return op.ReduceSum(X * X, keepdims=1)


@script(opset)
def square_loss(X, Y):
    return l2norm(X - Y)
