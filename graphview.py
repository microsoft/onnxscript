import onnxscript
import onnx
from onnxscript import ir
import onnxscript._internal

#load an onnx model

model = onnx.load("/home/t-assumange/llama2-7b_Dynamo_transformers4.41/myrules.onnx", load_external_data=False)

view = ir.from_proto(model)
# get the result and print it out
print("graph", view.functions[("pkg.transformers.4.41.2", "transformers_models_llama_modeling_llama_LlamaAttention_model_layers_10_self_attn_1", "")])

