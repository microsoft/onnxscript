# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx import opset15 as op


def NotEqual(left, right):
    return op.Not(op.Equal(left, right))
