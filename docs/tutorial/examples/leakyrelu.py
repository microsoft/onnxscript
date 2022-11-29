from onnxscript import script, opset15 as op


@script()
def LeakyRelu(X, alpha: float):
    return op.Where(X < 0.0, alpha * X, X)
