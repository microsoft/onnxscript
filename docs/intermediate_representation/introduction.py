# %% [markdown]
# # ONNX IR Introduction
# ## Overview
#
# The ONNX IR a robust, efficient and Pythonic in-memory IR for ONNX to power
# model building, analysis and manipulation. It has
#
# - **Full ONNX spec support**: all valid models representable by ONNX protobuf,
#   and a subset of invalid models (so you can load and fix them).
# - **Low memory footprint**: mmap'ed external tensors; unified interface for
#   ONNX TensorProto, Numpy arrays and PyTorch Tensors etc. No tensor size limitation.
#   Zero copies.
# - **Straightforward access patterns**: Access value information and traverse the
#   graph topology at ease.
# - **Robust mutation support**: Create as many iterators as you like on the graph
#   while mutating it.
# - **Speed**: Performant graph manipulation, serialization/deserialization to Protobuf.
# - **Pythonic and familiar APIs**: Classes define Pythonic apis and still map
#   to ONNX protobuf concepts in an intuitive way.
# - **No protobuf dependency**: The IR does not require protobuf once the model
#   is converted to the IR representation, decoupling from the serialization
#   format.
#
# This tutorial will demonstrate how you can use the ONNX IR to inspect, manipulate
# and build ONNX models.

# %%
import onnx
from onnxscript import ir

# %%
# Load ONNX model

proto = onnx.load("/home/justinchu/dev/onnx-script/testdata/e2e_models/mobilenetv2_100/dynamo/mobilenetv2_100_dynamo.onnx")
model = ir.serde.deserialize_model(proto)

# %%
model

# %%
graph = model.graph

# %%
graph.display()

# %%
print(graph.initializers.keys())

# %%
graph.initializers["conv_stem.weight"].display()

# %%
# graph.initializers["model.embed_tokens.weight"].numpy()

# %%
len(graph)

# %%
node = graph[6]

# %%
print(node)

# %%
new_node = ir.Node(
    "my_custom_domain",
    "Linear_classifier",
    node.inputs,
    name="new_torch_nn_modules_linear_Linear_classifier_1"
)

# %%
new_node.display()

# %%
for value, replacement in zip(node.outputs, new_node.outputs):
    for user_node, index in tuple(value.uses()):
        user_node.replace_input_with(index, replacement)

# %%
graph.insert_after(node, new_node)

# %%
print(graph)

# %%
graph.remove(node, safe=True)

# %%
graph.outputs[0] = new_node.outputs[0]

# %%
print(graph)

# %%
print(node.inputs)

# %%
graph.remove(node, safe=True)

# %%
new_model_proto = ir.serde.serialize_model(model)

# %%
graph.outputs[0].shape = ir.Shape([1, 1000])

# %%
graph.outputs[0].dtype = ir.DataType.FLOAT

# %%
graph.outputs[0]

# %%
print(graph)

# %%
new_model_proto = ir.serde.serialize_model(model)
onnx.save(new_model_proto, "new_model_proto.onnx")

# %%
len(node.inputs)

# %%
len(node.outputs)

# %%
graph.insert_after
