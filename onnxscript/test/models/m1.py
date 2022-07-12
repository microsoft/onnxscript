# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


def cadd(A: FLOAT[1, 2], B: FLOAT[1, 2]) -> FLOAT[1, 4]:
    return op.Concat(A, B, axis=-1)
