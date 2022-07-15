from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnx import helper, TensorProto

@script()
def tensor_attr(x):
    c = op.Constant(value = helper.make_tensor("scalar_half", TensorProto.FLOAT, (), [0.5]))
    return c * x
