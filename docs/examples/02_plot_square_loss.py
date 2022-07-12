"""
Generating a ModelProto
=======================

This example demonstrates the use of *onnx-script* to define an ONNX model.
*onnx-script* behaves like a compiler. It converts a script into an ONNX model.
"""

#%%
# First, we define the implementation of a square-loss function in onnxscript.

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

@script()
def square_loss(X: FLOAT["N", 1], Y: FLOAT["N", 1]) -> FLOAT[1, 1]:
    diff = X - Y
    return op.ReduceSum(diff * diff, keepdims=1)


#%%
# We can convert it to a model (an ONNX *ModelProto*) as follows:

model = square_loss.to_model_proto()

#%%
# Let's see what the generated model looks like.
from onnxscript.utils import proto_to_text
print(proto_to_text(model))

#%%
# We can run shape-inference and type-check the model using the standard ONNX API.
import onnx
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

#%%
# And finally, we can use *onnxruntime* to compute the outputs
# based on this model, using the standard onnxruntime API.
import numpy as np
from onnxruntime import InferenceSession

sess = InferenceSession(model.SerializeToString())

X = np.array([[0, 1, 2]], dtype=np.float32).T
Y = np.array([[0.1, 1.2, 2.3]], dtype=np.float32).T

got = sess.run(None, {'X': X, 'Y': Y})
expected = ((X - Y) ** 2).sum()

print(expected, got)
