# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.main import graph as graphattr
from onnxscript import script
from onnxscript.utils import proto2text
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, BOOL, INT64

import numpy as np

@script()
def cumulative_sum(X: INT64['N']):
    '''Test use of a nested-function as a graph-attribute, using the Scan operator.'''
    @graphattr(parent=cumulative_sum)
    def Sum(sum_in, next):
        sum_out = sum_in + next
        scan_out = op.Identity(sum_out)
        return sum_out, scan_out
    zero = op.Constant(value_int=0)
    all_sum, result = op.Scan (zero, X, body=Sum, num_scan_inputs=1)
    return result

@script()
def sum_to(X):
    '''Test use of a nested-function as a graph-attribute, using the Loop operator.'''
    @graphattr(parent=sum_to)
    def LoopBody(i: INT64, cond: BOOL, sum_in: INT64):
        cond_out = op.Identity(cond)
        sum_out = sum_in + i
        scan_out = op.Identity(sum_out)
        return cond_out, sum_out, scan_out
    zero = op.Constant(value_int=0)
    final_sum, result = op.Loop (X, None, zero, body=LoopBody)
    return result

