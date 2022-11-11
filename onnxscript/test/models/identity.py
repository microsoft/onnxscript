# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Test cases for automatic introduction of Identity (copy)

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, BOOL

@script()
def id1(A: FLOAT[...]) -> FLOAT[...]:
    return A

@script()
def id1_expanded(A: FLOAT[...]) -> FLOAT[...]:
    return op.Identity(A)

@script()
def id2(A: FLOAT[...]) -> FLOAT[...]:
    B = A
    return B

@script()
def id2_expanded(A: FLOAT[...]) -> FLOAT[...]:
    B = A
    return op.Identity(B)
