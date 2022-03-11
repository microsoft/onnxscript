"""
Use onnx-script to implement square loss function
=================================================

This example demonstrates *onnx-script* on one simple function.
*onnx-script* behaves like a compiler. It converts a script into
an ONNX graph.
"""
import numpy as np
import onnx
from onnxscript.converter import Converter
from onnxruntime import InferenceSession

#%%
# First the implementation of a square loss.

square_body = """
from onnxscript.onnx_types import FLOAT

def square_loss(X: FLOAT["N", 1], Y: FLOAT["N", 1]) -> FLOAT[1, 1]:
    diff = X - Y
    return oxs.ReduceSum(diff * diff, keepdims=1)
"""

#%%
# Then the compilation into an ONNX graph.

converter = Converter()
square_onnx = converter.convert(square_body)
graph = square_onnx[0].to_graph_proto()

#%%
# Let's see how the graph looks like.
print(onnx.helper.printable_graph(graph))

#%%
# In order to check the function does what it is expected,
# we create an ONNX model based on the code which was produced.
model = onnx.helper.make_model(
    graph, producer_name='onnx-script',
    opset_imports=[onnx.helper.make_opsetid("", 15)])
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

#%%
# And finally, we use *onnxruntime* to compute the outputs
# based on this graph.

sess = InferenceSession(model.SerializeToString())

X = np.array([[0, 1, 2]], dtype=np.float32).T
Y = np.array([[0.1, 1.2, 2.3]], dtype=np.float32).T

got = sess.run(None, {'X': X, 'Y': Y})
expected = ((X - Y) ** 2).sum()

print(expected, got)
