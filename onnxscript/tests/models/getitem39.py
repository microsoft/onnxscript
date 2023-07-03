# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT


@script()
def getitem_index_int(A: FLOAT[...]) -> FLOAT[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    r = A[scalar_zero : scalar_zero + 1, scalar_zero + 2]
    return r


@script()
def getitem_index_int2(A: FLOAT[...]) -> FLOAT[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    r = A[scalar_zero : scalar_zero + 1, 2]
    return r
