# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT

@script(default_opset=op)
def getitem_i(A: FLOAT[...]) -> FLOAT[...]:
    return A[0]

@script(default_opset=op)
def getitem_i_last(A: FLOAT[...]) -> FLOAT[...]:
    return A[-1]

@script(default_opset=op)
def getitem_column(A: FLOAT[...]) -> FLOAT[...]:
    return A[:, 1]

@script()
def getitem_index_int0(A: FLOAT[...]) -> FLOAT[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero]

@script()
def getitem_index_int0_1(A: FLOAT[...]) -> FLOAT[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1]

@script(default_opset=op)
def getitem_i_slice(A: FLOAT[...]) -> FLOAT[...]:
    return A[1:2]

@script(default_opset=op)
def getitem_i_slice_left(A: FLOAT[...]) -> FLOAT[...]:
    return A[1:]

@script(default_opset=op)
def getitem_i_slice_right(A: FLOAT[...]) -> FLOAT[...]:
    return A[:2]

@script(default_opset=op)
def getitem_i_slice_neg(A: FLOAT[...]) -> FLOAT[...]:
    return A[1:-1]

@script(default_opset=op)
def getitem_rev(A: FLOAT[...]) -> FLOAT[...]:
    return A[:0:-1]

@script(default_opset=op)
def getitem_rev0(A: FLOAT[...]) -> FLOAT[...]:
    return A[0, :0:-1]

@script(default_opset=op)
def getitem_i_mixed_tuple(A: FLOAT[...]) -> FLOAT[...]:
    return A[:2, 0]

@script(default_opset=op)
def getitem_i_tuple(A: FLOAT[...]) -> FLOAT[...]:
    return A[:2, :1]

@script(default_opset=op)
def getitem_i_slice_step(A: FLOAT[...]) -> FLOAT[...]:
    return A[2:0:-1]

@script()
def getitem_i_var(A: FLOAT[...]) -> FLOAT[...]:
    # eager mode does not work on this one:
    # TypeError: only integer scalar arrays can be converted to a scalar index
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1 : scalar_zero + 2]

@script(default_opset=op)
def getitem_i_expr(A: FLOAT[...]) -> FLOAT[...]:
    r = (A + 1)[0]
    return r


# This notation is not possible with ONNX but is allowed by numpy.
# @script()
# def getitem_i_slice_right_step(A: FLOAT[...]) -> FLOAT[...]:
#     r = A[1::-1]
#     return r
