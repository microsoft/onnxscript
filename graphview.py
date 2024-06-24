import onnxscript
import onnx

import onnxscript._internal

#load an onnx model

model = onnx.load("/home/t-assumange/llama3-8-Bdynamo/without_in.onnx", load_external_data=False)

view = onnxscript.proto2python(model)
# get the result and print it out
print("graph", view)


