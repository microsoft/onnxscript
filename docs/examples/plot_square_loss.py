"""
Overview: A model implementing square-loss
=================================================

This example demonstrates *onnx-script* on one simple function.
*onnx-script* behaves like a compiler. It converts a script into
an ONNX graph.
"""

#%%
# First the implementation of a square loss.

from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT

@script()
def square_loss(X: FLOAT["N", 1], Y: FLOAT["N", 1]) -> FLOAT[1, 1]:
    diff = X - Y
    return op.ReduceSum(diff * diff, keepdims=1)


#%%
# Convert it to a model.

model = square_loss.to_model_proto()

#%%
# Let's see how the graph looks like.
import onnx
import onnx.printer
print(onnx.printer.to_text(model))

#%%
# Check the model

model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

#%%
# And finally, we use *onnxruntime* to compute the outputs
# based on this graph.
import numpy as np
from onnxruntime import InferenceSession

sess = InferenceSession(model.SerializeToString())

X = np.array([[0, 1, 2]], dtype=np.float32).T
Y = np.array([[0.1, 1.2, 2.3]], dtype=np.float32).T

got = sess.run(None, {'X': X, 'Y': Y})
expected = ((X - Y) ** 2).sum()

print(expected, got)
