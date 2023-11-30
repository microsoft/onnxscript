"""
Generating a LibProto
=====================

The examples below show how we can define a library consisting of multiple functions,
and export it.

**This is preliminary. Proto extensions are required to fully support LibProto.**
"""

from onnxscript import export_onnx_lib, script
from onnxscript import opset15 as op
from onnxscript.values import Opset

# %%
# The domain/version of the library functions defined below
opset = Opset("com.mydomain", 1)


# %%
# The definitions of the functions:
@script(opset)
def l2norm(X):
    return op.ReduceSum(X * X, keepdims=1)


@script(opset)
def square_loss(X, Y):
    return l2norm(op.Sub(X, Y))


# %%
# Export the functions as an ONNX library.
export_onnx_lib([l2norm, square_loss], "mylib.onnxlib")
