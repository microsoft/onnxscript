# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Test cases for automatic introduction of Identity (copy)

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import BOOL, FLOAT, INT64


@script(default_opset=op)
def id1(A: FLOAT[...]) -> FLOAT[...]:
    return A  # treat as op.Identity(A)


@script()
def id1_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return op.Identity(A)


@script(default_opset=op)
def id2(A: FLOAT[...]) -> FLOAT[...]:
    B = A
    return B  # treat as op.Identity(B) == op.Identity(A)


@script()
def id2_expanded(A: FLOAT[...]) -> FLOAT[...]:
    B = A
    return op.Identity(B)


@script()
def control_flow_id1(A: FLOAT[...], flag: BOOL) -> FLOAT[...]:
    if flag:
        y = A  # treat as op.Identity(A)
    else:
        y = op.Abs(A)
    return y


@script()
def control_flow_id1_expanded(A: FLOAT[...], flag: BOOL) -> FLOAT[...]:
    if flag:
        y = op.Identity(A)
    else:
        y = op.Abs(A)
    return y


@script()
def loop_id(A: FLOAT[...], N: INT64) -> FLOAT[...]:
    B = op.Identity(A)
    for i in range(N):
        B = A  # treat as op.Identity(A)
        A = A + 1
    return A + B


@script()
def loop_id_expanded(A: FLOAT[...], N: INT64) -> FLOAT[...]:
    B = op.Identity(A)
    for i in range(N):
        B = op.Identity(A)
        A = A + 1
    return A + B
