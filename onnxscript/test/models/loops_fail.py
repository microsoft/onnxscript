# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, conditional_range
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

# same variable assigned multiple times


@script()
def loop_fail(A: FLOAT["N"]) -> FLOAT["N"]:
    T = A
    cond = op.Constant(value=make_tensor('true', TensorProto.BOOL, [1], [1]))
    for i in conditional_range(10, cond):
        T = T + A * op.Cast(i, to=TensorProto.FLOAT)
        pos = op.ReduceSum(T) > -10
        if pos:
            cond &= not pos
        else:
            cond &= pos
    return T
