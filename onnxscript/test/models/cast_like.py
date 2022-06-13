# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT, BOOL


@script()
def inc_right(A: FLOAT[...]) -> FLOAT[...]:
    return A + 1


@script()
def inc_left(A: FLOAT[...]) -> FLOAT[...]:
    return 1 + A


@script()
def cmp_zero_right(A: FLOAT[...]) -> BOOL[...]:
    return A == 0


@script()
def cmp_zero_left(A: FLOAT[...]) -> BOOL[...]:
    return 0 == A
