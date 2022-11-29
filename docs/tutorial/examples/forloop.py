from onnxscript import script, opset15 as op


@script()
def sumprod(x, N):
    sum = op.Identity(x)
    prod = op.Identity(x)
    for i in range(N):
        sum = sum + x
        prod = prod * x
    return sum, prod
