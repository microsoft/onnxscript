from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset16 as op16

@script(Opset('this', 2))
def Selu(X, alpha: float = 1.67326319217681884765625,
         gamma: float = 1.05070102214813232421875):
    zero = op16.Constant(value_float=1.)
    # neg = gamma * (alpha * op16.Exp(X) - alpha)
    neg = op16.Mul(gamma, op16.Sub(op16.Mul(alpha, op16.Exp(X)), alpha))
    # pos = gamma * X
    pos = op16.Sub(gamma, X)
    x_less_than_zero = op16.Less(X, zero)
    return op16.Where(x_less_than_zero, neg, pos)
