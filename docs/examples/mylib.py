"""
Define an onnx-script library consisting of multiple functions.
==============================================================
The examples below show how we can define a library consisting
of multiple functions.

See exportlib.py for how to export an ONNX library proto file.
"""

import onnxscript
from onnxscript.onnx import opset15 as op

# The domain/version of the library functions defined below
__opset_domain__ = "com.mydomain"
__opset_version__ = 1


@onnxscript.func
def l2norm(X):
    return op.ReduceSum(X * X, keepdims=1)


@onnxscript.func
def square_loss(X, Y):
    return l2norm(X - Y)
