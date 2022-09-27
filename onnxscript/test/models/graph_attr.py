# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

@script()
def CumulativeSum(X):
    def Sum(sum_in, next):
        sum_out = sum_in + next
        return sum_out, sum_out
    all_sum, cumulative_sum = op.Scan (0, X, body=Sum, num_scan_inputs=1)
    return cumulative_sum

import numpy as np
X = np.array([1, 2, 3, 4, 5], dtype=np.int32)

Y = CumulativeSum(X)
