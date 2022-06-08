# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript.onnx_types import DOUBLE, INT64
from onnxscript import script
from onnxscript.onnx import opset15 as op


@script()
def double_abs_subgraph(A: DOUBLE["N"]) -> DOUBLE["N"]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.DOUBLE, [1], [0]))
    if op.ReduceSum(A) > zero:
        B: DOUBLE["N"] = op.Identity(A)
    else:
        B: DOUBLE["N"] = op.Neg(A)
    return B


@script()
def double_abs(A: DOUBLE["N"]) -> DOUBLE["N"]:
    return op.Abs(A)


@script()
def double_cast(A: INT64["N"]) -> DOUBLE["N"]:
    return op.Cast(A, to=11)


# Does not work.
# @script()
# def double_abs_subgraph_direct_return(A: DOUBLE["N"]) -> DOUBLE["N"]:
#     zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
#     if op.ReduceSum(A) > zero:
#         return op.Identity(A)
#     else:
#         return op.Neg(A)
