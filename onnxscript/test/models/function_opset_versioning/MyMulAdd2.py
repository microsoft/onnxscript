from onnxscript import script
from onnxscript.onnx import opset16 as op16
from onnxscript.values import Opset

@script(Opset('this', 2))
def MyMul(X, A):
    return op16.Mul(X, A)

@script(Opset('this', 2))
def MyAdd(X, B):
    return op16.Add(X, B)

@script(Opset('this', 2))
def MyMulAdd(X, A, B):
    tmp = MyMul(X, A)
    return MyAdd(tmp, B)

@script(Opset('this', 2))
def MyWhere(X):
    zero = op16.Constant(value_float=0.)
    return op16.Where(X > zero, X, zero)
