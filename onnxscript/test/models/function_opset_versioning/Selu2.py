from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset16 as op16

@script(Opset('this', 2))
def Selu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op16.Constant(value_float=1.)
    neg = gamma * (alpha * op16.Exp(X) - alpha)
    pos = gamma * X
    return op16.Where(X <= zero, neg, pos)
