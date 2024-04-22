from onnxscript import script
from onnxscript.onnx_opset import opset17
from onnxscript.values import Opset

my__domain__com1 = Opset("my.domain.com", 1)


@script(my__domain__com1)
def twice(X):
    return_val = opset17.Add(X, X)
    return return_val
