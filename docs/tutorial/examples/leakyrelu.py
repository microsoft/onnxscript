from onnxscript import script
from onnxscript.onnx_opset import opset15 as op

@script()
def LeakyRelu(X, alpha: float):
    return op.Where(X < 0.0, alpha * X, X)