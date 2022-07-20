from onnxscript import script
from onnxscript.onnx import opset15 as op15
from onnxscript.values import Opset

@script(Opset('this', 1))
def MyMul(X, A):
    return op15.Mul(X, A)

@script(Opset('this', 1))
def MyAdd(X, B):
    return op15.Add(X, B)

@script(Opset('this', 1))
def MyMulAdd(X, A, B):
    tmp = MyMul(X, A)
    return MyAdd(tmp, B)

@script(Opset('this', 1))
def MyWhere(X):
    zero = op15.Constant(value_float=0.)
    return op15.Where(X > zero, X, zero)
