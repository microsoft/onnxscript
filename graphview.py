import onnxscript
import onnx
from onnxscript import ir
import onnxscript._internal

#load an onnx model

model = onnx.load("/home/t-assumange/llama3-8-Bdynamo_transformers4.41/consultation.onnx", load_external_data=False)

view = ir.from_proto(model)
# get the result and print it out
# see = view.functions[("pkg.transformers.4.41.2", "transformers_models_llama_modeling_llama_LlamaRotaryEmbedding_model_layers_9_self_attn_rotary_emb_1", "")]
print("graph", view.graph)

# print the graph and save it to an output.txt file in the current directory


# Print the graph
# print(view.graph) to see whole graph

# Save the graph to an output.txt file in the current directory
with open("consultation.txt", "w") as f:
    # f.write(str(view.graph))
    f.write(str(view.graph))