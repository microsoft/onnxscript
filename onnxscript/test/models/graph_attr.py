# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import graph, script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import BOOL, INT64


@script()
def cumulative_sum(X: INT64['N']):
    '''Test use of a nested-function as a graph-attribute, using the Scan operator.'''
    @graph(parent=cumulative_sum)
    def Sum(sum_in, next):
        sum_out = sum_in + next
        scan_out = op.Identity(sum_out)
        return sum_out, scan_out
    zero = op.Constant(value_int=0)
    _, result = op.Scan (zero, X, body=Sum, num_scan_inputs=1)
    return result

@script()
def sum_to(X: INT64):
    '''Test use of a nested-function as a graph-attribute, using the Loop operator.'''
    @graph(parent=sum_to)
    def LoopBody(i: INT64, cond: BOOL, sum_in: INT64):
        cond_out = op.Identity(cond)
        sum_out = sum_in + i
        scan_out = op.Identity(sum_out)
        return cond_out, sum_out, scan_out
    zero = op.Constant(value_int=0)
    _, result = op.Loop (X, None, zero, body=LoopBody)
    return result
