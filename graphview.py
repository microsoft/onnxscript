import onnxscript
import onnx
from onnxscript import ir
import onnxscript._internal

#load an onnx model

model = onnx.load("/home/t-assumange/llama2-7b/optimize_model_llama3.onnx", load_external_data=False)

view = ir.from_proto(model)
# get the result and print it out
print("graph", view.functions[("pkg.torch.2.3.1+cu121", "torch_nn_modules_linear_Linear_model_layers_10_mlp_down_proj_1", "")])

