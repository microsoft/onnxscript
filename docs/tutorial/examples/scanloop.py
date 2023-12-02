from onnxscript import graph, script
from onnxscript import opset15 as op


@script()
def CumulativeSum(X):
    @graph()
    def Sum(sum_in, next):
        sum_out = sum_in + next
        return sum_out, sum_out

    all_sum, cumulative_sum = op.Scan(0, X, body=Sum, num_scan_inputs=1)
    return cumulative_sum
