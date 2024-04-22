from onnxscript import script
from onnxscript.onnx_opset import opset17
from onnxscript.values import Opset

this1 = Opset("this", 1)


@script(this1)
def fun_with_attr_param(X, dtype: int):
    return_val = opset17.Cast(X, to=dtype)
    return return_val
