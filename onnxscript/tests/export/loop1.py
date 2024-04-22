from onnxscript import script
from onnxscript.onnx_opset import opset17
from onnxscript.values import Opset

this1 = Opset("this", 1)


@script(this1)
def loop1(X, N):
    Sum = opset17.Identity(X)
    Sum_0 = Sum
    for i in range(N):
        Sum_1 = opset17.Add(Sum_0, X)
        Sum_0 = Sum_1
    Sum_2 = Sum_0
    return Sum_2
