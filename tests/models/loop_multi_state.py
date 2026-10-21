# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64

# Five loop-carried state variables: enough for the converter's output-name
# ordering (built via a set union) to diverge from its input ordering (built
# by iterating the same set directly) under CPython's hash randomization.


@script()
def loop_multi_state(
    x: FLOAT["N"],
    n: INT64,  # noqa: F821
) -> (FLOAT["N"], FLOAT["N"], FLOAT["N"], FLOAT["N"], FLOAT["N"]):  # noqa: F821
    a = op.Identity(x)
    b = op.Identity(x)
    c = op.Identity(x)
    d = op.Identity(x)
    e = op.Identity(x)
    i = 0
    cond = True
    while cond:
        a = op.Add(a, x)
        b = op.Mul(b, x)
        c = op.Sub(c, x)
        d = op.Div(d, x)
        e = op.Add(e, b)
        i = i + 1
        cond = i < n
    return a, b, c, d, e
