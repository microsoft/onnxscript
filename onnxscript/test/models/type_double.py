# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import DOUBLE, INT64


@script()
def sumprod_typed(x: DOUBLE["N"], N: INT64) -> DOUBLE["N"]:  # noqa: F821
    weight = op.Constant(value=make_tensor("zero", TensorProto.DOUBLE, [1], [0]))
    weighted_sum = op.Identity(weight)
    for i in range(N):
        b: DOUBLE[...] = op.Cast(i, to=11)
        a: DOUBLE[...] = x * b
        weighted_sum = weighted_sum + a
        weight = weight + b
    return op.Div(weighted_sum, weight)


@script()
def sumprod(x: DOUBLE["N"], N: INT64) -> DOUBLE["N"]:  # noqa: F821
    weight = op.Constant(value=make_tensor("zero", TensorProto.DOUBLE, [1], [0]))
    weighted_sum = op.Identity(weight)
    for i in range(N):
        b = op.Cast(i, to=11)
        a = x * b
        weighted_sum = weighted_sum + a
        weight = weight + b
    return op.Div(weighted_sum, weight)


@script()
def notype_abs_subgraph(A):
    zero = op.Constant(value=make_tensor("zero", TensorProto.FLOAT, [1], [0]))
    if op.Cast(op.ReduceSum(A), to=1) > zero:
        B = op.Identity(A)
    else:
        B = op.Neg(A)
    return B


@script()
def double_abs_subgraph(A: DOUBLE["N"]) -> DOUBLE["N"]:
    zero = op.Constant(value=make_tensor("zero", TensorProto.DOUBLE, [1], [0]))
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
