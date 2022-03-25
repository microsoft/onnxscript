# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT, INT64

# simple loop


def sumprod(x: FLOAT['N'], N: INT64) -> (FLOAT['N'], FLOAT['N']):
    """
    Combines ReduceSum, ReduceProd.
    """
    sum = oxs.Identity(x)
    prod = oxs.Identity(x)
    for i in range(N):
        sum = sum + x
        prod = prod * x
    return sum, prod
