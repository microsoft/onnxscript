import onnxscript
import onnx
from onnxscript import ir
import onnxscript._internal

#load an onnx model

model = onnx.load("/home/t-assumange/llama2-7b_Dynamo_transformers4.41/myrules.onnx", load_external_data=False)

view = ir.from_proto(model)
# get the result and print it out
# print("graph", view.functions[("pkg.transformers.4.41.2", "transformers_models_llama_modeling_llama_LlamaAttention_model_layers_0_self_attn_1", "")])
# print("graph", view.graph)
# print the graph and save it to an output.txt file in the current directory


# Print the graph
print(view.graph)

# Save the graph to an output.txt file in the current directory
with open("output.txt", "w") as f:
    f.write(str(view.graph))