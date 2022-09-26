# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.main import graph as graphattr
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, BOOL, INT64

import numpy as np

@script()
def CumulativeSum(X):
    '''Test use of a nested-function as a graph-attribute, using the Scan operator.'''
    @graphattr(parent=CumulativeSum)
    def Sum(sum_in, next):
        sum_out = sum_in + next
        scan_out = op.Identity(sum_out)
        return sum_out, scan_out
    all_sum, cumulative_sum = op.Scan (0, X, body=Sum, num_scan_inputs=1)
    return cumulative_sum

model = CumulativeSum.to_model_proto()

# X = np.array([1, 2, 3, 4, 5], dtype=np.int32)
# Y = CumulativeSum(X)
# print(Y)

@script()
def SumTo(N):
    '''Test use of a nested-function as a graph-attribute, using the Loop operator.'''
    @graphattr(parent=SumTo)
    def LoopBody(i: INT64, cond: BOOL, sum_in: INT64):
        cond_out = op.Identity(cond)
        sum_out = sum_in + i
        scan_out = op.Identity(sum_out)
        return cond_out, sum_out, scan_out
    zero = op.Constant(value_int=0)
    all_sum, cumulative_sum = op.Loop (N, None, zero, body=LoopBody)
    return cumulative_sum

X = np.array([5], dtype = np.int64).reshape(())
Y = SumTo(X)
print(Y)
