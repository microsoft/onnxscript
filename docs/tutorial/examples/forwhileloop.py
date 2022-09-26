from onnxscript import script
from onnxscript.onnx_opset import opset15 as op


@script()
def sumprod_break(x, N):
    sum = op.Identity(x)
    prod = op.Identity(x)
    for i in range(N):
        sum = sum + x
        prod = prod * x
        cond = op.ReduceSum(prod) > 1e7
        # ONNX does not support break instruction.
        # onnxscript can only convert example if the break
        # instruction is placed at the end of the loop body.
        if cond:
            break
    return sum, prod
