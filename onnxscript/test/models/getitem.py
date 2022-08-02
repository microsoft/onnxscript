# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def getitem_i_last(A: FLOAT[...]) -> FLOAT[...]:
    r = A[-1]
    return r


@script()
def getitem_i(A: FLOAT[...]) -> FLOAT[...]:
    r = A[0]
    return r
