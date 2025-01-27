# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def sumprod_break(x, N):
    sum = op.Identity(x)
    prod = op.Identity(x)
    for _ in range(N):
        sum = sum + x
        prod = prod * x
        cond = op.ReduceSum(prod) > 1e7
        # ONNX does not support break instruction.
        # onnxscript can only convert example if the break
        # instruction is placed at the end of the loop body.
        if cond:
            break
    return sum, prod
