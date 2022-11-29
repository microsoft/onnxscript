from onnxscript import script, opset15 as op


@script()
def FirstDim(X):
    return op.Shape(X, start=0, end=1)
