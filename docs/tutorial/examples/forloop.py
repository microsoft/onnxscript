# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def sumprod(x, N):
    sum = op.Identity(x)
    prod = op.Identity(x)
    for _ in range(N):
        sum = sum + x
        prod = prod * x
    return sum, prod
