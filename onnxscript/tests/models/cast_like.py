# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Test cases for automatic introduction of CastLike around constants:

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import BOOL, FLOAT


@script(default_opset=op)
def inc_right(A: FLOAT[...]) -> FLOAT[...]:
    return A + 1


@script()
def inc_right_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return A + op.CastLike(1, A)


@script(default_opset=op)
def inc_left(A: FLOAT[...]) -> FLOAT[...]:
    return 1 + A


@script()
def inc_left_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return op.CastLike(1, A) + A


@script(default_opset=op)
def cmp_zero_right(A: FLOAT[...]) -> BOOL[...]:
    return A == 0


@script()
def cmp_zero_right_expanded(A: FLOAT[...]) -> BOOL[...]:
    return A == op.CastLike(0, A)


@script(default_opset=op)
def cmp_zero_mright(A: FLOAT[...]) -> BOOL[...]:
    return A == -11


@script()
def cmp_zero_mright_expanded(A: FLOAT[...]) -> BOOL[...]:
    return A == op.CastLike(-11, A)


@script(default_opset=op)
def cmp_zero_left(A: FLOAT[...]) -> BOOL[...]:
    return 0 == A


@script()
def cmp_zero_left_expanded(A: FLOAT[...]) -> BOOL[...]:
    return op.CastLike(0, A) == A


@script(default_opset=op)
def div_right(A: FLOAT[...]) -> FLOAT[...]:
    return A / 2


@script()
def div_right_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return A / op.CastLike(2, A)


@script(default_opset=op)
def div_minus_right(A: FLOAT[...]) -> FLOAT[...]:
    return A / (-2)


@script()
def div_minus_right_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return A / op.CastLike(-2, A)


# @script()
# def div_minus_minus_right(A: FLOAT[...]) -> FLOAT[...]:
#     return A / (-(-2))


@script()
def where_left(C: BOOL[...], A: FLOAT[...]) -> FLOAT[...]:
    return op.Where(C, 2, A)


@script()
def where_left_expanded(C: BOOL[...], A: FLOAT[...]) -> FLOAT[...]:
    return op.Where(C, op.CastLike(2, A), A)


@script()
def where_right(C: BOOL[...], A: FLOAT[...]) -> FLOAT[...]:
    return op.Where(C, A, 3)


@script()
def where_right_expanded(C: BOOL[...], A: FLOAT[...]) -> FLOAT[...]:
    return op.Where(C, A, op.CastLike(3, A))
