# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT


def multi(A: FLOAT["N"]) -> FLOAT["N"]:
    x, y = op.Split(A)
    x, y = op.Split(x)
    return x + y
