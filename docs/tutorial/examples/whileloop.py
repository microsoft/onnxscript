from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op


@script()
def sumprod(x, N):
    sum = op.Identity(x)
    prod = op.Identity(x)
    cond = op.Constant(value=make_tensor('true', TensorProto.BOOL, [1], [1]))
    i = op.Constant(value=make_tensor('i', TensorProto.INT64, [1], [0]))
    while cond:
        sum = sum + x
        prod = prod * x
        i = i + 1
        cond = i < 10
    return sum, prod
