# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import graph, script
from onnxscript import opset15 as op

try:

    @script()
    def CumulativeSum(X):
        g = op.Constant(value=0)

        @graph()
        def Sum(sum_in, next):
            sum_out = sum_in + next + g
            return sum_out, sum_out

        g = op.Constant(value=1)
        _all_sum, cumulative_sum = op.Scan(0, X, body=Sum, num_scan_inputs=1)
        return cumulative_sum

except Exception as e:
    assert "Outer scope variable" in str(e)
