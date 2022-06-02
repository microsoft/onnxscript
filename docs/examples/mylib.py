"""
Define an onnx-script library consisting of multiple functions,
and export it to ONNX proto format.
==============================================================
The examples below show how we can define a library consisting
of multiple functions.
"""

from onnxscript import script, export_onnx_lib
from onnxscript.onnx import opset15 as op
from onnxscript.values import CustomOpset

# The domain/version of the library functions defined below
opset = CustomOpset('com.mydomain', 1)


@script(opset)
def l2norm(X):
    return op.ReduceSum(X * X, keepdims=1)


@script(opset)
def square_loss(X, Y):
    return l2norm(X - Y)

# Export the functions as an ONNX library.
export_onnx_lib([l2norm, square_loss], "mylib.onnxlib")
