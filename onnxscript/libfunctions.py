from onnxscript.onnx import opset15 as op


def NotEqual(left, right):
    return op.Not(op.Equal(left, right))
