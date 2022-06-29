"""
Exporting a library of multiple functions
===================================================

The examples below show how we can define a library consisting
of multiple functions.

**This is preliminary. Proto extensions required to fully support LibProto.**
"""

from onnxscript import script, export_onnx_lib
from onnxscript.onnx import opset15 as op
from onnxscript.values import Opset

#%%
# The domain/version of the library functions defined below
opset = Opset('com.mydomain', 1)

#%%
# The definitions of the functions:
@script(opset)
def l2norm(X):
    return op.ReduceSum(X * X, keepdims=1)


@script(opset)
def square_loss(X, Y):
    return l2norm(X - Y)

#%%
# Export the functions as an ONNX library.
export_onnx_lib([l2norm, square_loss], "mylib.onnxlib")
