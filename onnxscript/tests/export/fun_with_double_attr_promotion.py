from onnxscript import script
from onnxscript.onnx_opset import opset17
from onnxscript.values import Opset

this1 = Opset("this", 1)


@script(this1)
def fun_with_double_attr_promotion(X, dtype: int):
    dtype_1 = opset17.Constant(value_int=dtype)
    dtype_cast = opset17.CastLike(dtype_1, X)
    Y = opset17.Add(X, dtype_cast)
    dtype_0 = opset17.Constant(value_int=dtype)
    dtype_0_cast = opset17.CastLike(dtype_0, Y)
    Z = opset17.Add(Y, dtype_0_cast)
    return Z
