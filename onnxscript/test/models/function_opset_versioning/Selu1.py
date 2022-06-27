from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15

@script(Opset('this', 1))
def Selu(X: FLOAT[None], alpha: FLOAT[1], gamma: FLOAT[1]) -> FLOAT[None]:
    zero = op15.Constant(value_float=1.)
    neg = gamma * (alpha * op15.Exp(X) - alpha)
    pos = gamma * X
    return op15.Where(X <= zero, neg, pos)
