import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_opset import opset17
my__domain__com1 = Opset('my.domain.com', 1)

@script(my__domain__com1)
def twice(X):
    return_val = opset17.Add(X, X)
    return return_val
