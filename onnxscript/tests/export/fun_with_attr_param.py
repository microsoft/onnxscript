
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_opset import opset17

this1 = Opset("this", 1)

@script(this1)
def fun_with_attr_param(X, dtype: int):
    
    return_val = opset17.Cast(X, to=dtype)
    return return_val


