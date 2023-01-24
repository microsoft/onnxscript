# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script(default_opset=op)
def eager_op(X: FLOAT[...]) -> FLOAT[...]:
    return X % 1.5


@script()
def eager_abs(X: FLOAT[...]) -> FLOAT[...]:
    return op.Abs(X) + 1
