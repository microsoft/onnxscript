# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT


def nested_if(A: FLOAT["N"], B: FLOAT["N"], C: FLOAT['N']) -> FLOAT["N"]:
    sum1 = oxs.ReduceSum(A)
    sum2 = oxs.ReduceSum(B)
    sum3 = oxs.ReduceSum(C)
    if sum1 < sum2:
        if sum1 < sum3:
            result = oxs.Identity(A)
        else:
            result = oxs.Identity(B)
    else:
        result = oxs.Identity(C)
    return result
