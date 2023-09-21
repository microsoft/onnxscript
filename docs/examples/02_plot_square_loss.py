"""
Generating a ModelProto
=======================

This example demonstrates the use of *onnxscript* to define an ONNX model.
*onnxscript* behaves like a compiler. It converts a script into an ONNX model.

"""

# %%
# First, we define the implementation of a square-loss function in onnxscript.

import numpy as np
import onnx
from onnxruntime import InferenceSession

from onnxscript import FLOAT
from onnxscript import opset15 as op
from onnxscript import proto2text, script


@script()
def square_loss(X: FLOAT["N", 1], Y: FLOAT["N", 1]) -> FLOAT[1, 1]:  # noqa: F821
    diff = X - Y
    return op.ReduceSum(diff * diff, keepdims=1)


# %%
# We can convert it to a model (an ONNX *ModelProto*) as follows:

model = square_loss.to_model_proto()

# %%
# Let's see what the generated model looks like.
print(proto2text(model))

# %%
# We can run shape-inference and type-check the model using the standard ONNX API.

model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

# %%
# And finally, we can use *onnxruntime* to compute the outputs
# based on this model, using the standard onnxruntime API.

sess = InferenceSession(model.SerializeToString(), providers=("CPUExecutionProvider",))

X = np.array([[0, 1, 2]], dtype=np.float32).T
Y = np.array([[0.1, 1.2, 2.3]], dtype=np.float32).T

got = sess.run(None, {"X": X, "Y": Y})
expected = ((X - Y) ** 2).sum()

print(expected, got)
