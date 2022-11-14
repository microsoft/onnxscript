"""
Eager mode evaluation
=====================

An *onnx-script* function can be executed directly as a Python function (for example,
with a Python debugger). This is useful for debugging an *onnx-script* function definition.
This execution makes use of a backend implementation of the ONNX ops used in the function
definition. Currently, the backend implementation uses onnxruntime to execute each op
invocation. This mode of execution is referred to as *eager mode evaluation*.

The example below illustrates this. We first define an *onnx-script* function:
"""
import numpy as np

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def linear(
    A: FLOAT["N", "K"], W: FLOAT["K", "M"], Bias: FLOAT["M"]
) -> FLOAT["N", "M"]:  # noqa: F821
    T1 = op.MatMul(A, W)
    T2 = op.Add(T1, Bias)
    Y = op.Relu(T2)
    return Y


#%%
# Create inputs for evaluating the function:

np.random.seed(0)
m = 4
k = 16
n = 4
a = np.random.rand(k, m).astype("float32").T
w = np.random.rand(n, k).astype("float32").T
b = np.random.rand(n).astype("float32").T

#%%
# Evaluate the function:
print(linear(a, w, b))
