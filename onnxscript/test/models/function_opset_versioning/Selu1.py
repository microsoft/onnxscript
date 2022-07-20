from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15

@script(Opset('this', 1))
def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    zero = op15.Constant(value_float=1.)
    neg = gamma * (alpha * op15.Exp(X) - alpha)
    pos = gamma * X
    return op15.Where(X <= zero, neg, pos)
