# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64

# simple loop


def sumprod(x: FLOAT['N'], N: INT64) -> (FLOAT['N'], FLOAT['N']):
    """
    Combines ReduceSum, ReduceProd.
    """
    sum = op.Identity(x)
    prod = op.Identity(x)
    for i in range(N):
        sum = sum + x
        prod = prod * x
    return sum, prod
