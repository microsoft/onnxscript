# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import INT32
from onnxscript.tests.common.onnx_script_test_case import FunctionTestParams as Test

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)

@script(default_opset=op)
def getitem_i(A: INT32[...]) -> INT32[...]:
    return A[0]

get_first_row_test = Test(getitem_i, input=[x], output=[np.array([0, 1, 2], dtype=np.int32)])

@script(default_opset=op)
def getitem_i_last(A: INT32[...]) -> INT32[...]:
    return A[-1]

get_last_row_test = Test(getitem_i_last, input=[x], output=[np.array([9, 10, 11], dtype=np.int32)])

@script(default_opset=op)
def getitem_column(A: INT32[...]) -> INT32[...]:
    return A[:, 1]

get_col_1_test = Test(getitem_column, input=[x], output=[np.array([1, 4, 7, 10], dtype=np.int32)])

@script()
def getitem_index_int0(A: INT32[...]) -> INT32[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero]

computed_row_test = Test(getitem_index_int0, input=[x], output=[np.array([0, 1, 2], dtype=np.int32)])

@script()
def getitem_index_int0_1(A: INT32[...]) -> INT32[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1]

computed_row_test2 = Test(getitem_index_int0_1, input=[x], output=[np.array([3, 4, 5], dtype=np.int32)])

@script(default_opset=op)
def getitem_i_slice(A: INT32[...]) -> INT32[...]:
    return A[1:2]

slice_from_to = Test(getitem_i_slice, input=[x], output=[np.array([[3, 4, 5]], dtype=np.int32)])

@script(default_opset=op)
def getitem_i_slice_left(A: INT32[...]) -> INT32[...]:
    return A[1:]

slice_from = Test(getitem_i_slice_left, input=[x], output=[np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)])

@script(default_opset=op)
def getitem_i_slice_right(A: INT32[...]) -> INT32[...]:
    return A[:2]

slice_to = Test(getitem_i_slice_right, input=[x], output=[np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)])

@script(default_opset=op)
def getitem_i_slice_neg(A: INT32[...]) -> INT32[...]:
    return A[1:-1]

slice_from_neg_to = Test(getitem_i_slice_neg, input=[x], output=[np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int32)])

@script(default_opset=op)
def getitem_rev(A: INT32[...]) -> INT32[...]:
    return A[:0:-1]

slice_to_neg_step = Test(getitem_rev, input=[x], output=[x[:0:-1]])

@script(default_opset=op)
def getitem_rev0(A: INT32[...]) -> INT32[...]:
    return A[0, :0:-1]

index_and_slice = Test(getitem_rev0, input=[x], output=[x[0, :0:-1]])

@script(default_opset=op)
def getitem_i_mixed_tuple(A: INT32[...]) -> INT32[...]:
    return A[:2, 0]

slice_and_index = Test(getitem_i_mixed_tuple, input=[x], output=[x[:2, 0]])

@script(default_opset=op)
def getitem_i_tuple(A: INT32[...]) -> INT32[...]:
    return A[:2, :1]

slice_and_slice = Test(getitem_i_tuple, input=[x], output=[x[:2, :1]])

@script(default_opset=op)
def getitem_i_slice_step(A: INT32[...]) -> INT32[...]:
    return A[2:0:-1]

slice_from_to_neg_step = Test(getitem_i_slice_step, input=[x], output=[x[2:0:-1]])

@script()
def getitem_i_var(A: INT32[...]) -> INT32[...]:
    # eager mode does not work on this one:
    # TypeError: only integer scalar arrays can be converted to a scalar index
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1 : scalar_zero + 2]

slice_computed_range = Test(getitem_i_var, input=[x], output=[x[1:2]])

@script(default_opset=op)
def getitem_i_expr(A: INT32[...]) -> INT32[...]:
    r = (A + 1)[0]
    return r

index_computed_tensor = Test(getitem_i_expr, input=[x], output=[np.array([1, 2, 3], dtype=np.int32)])

# This notation is not possible with ONNX but is allowed by numpy.
# @script()
# def getitem_i_slice_right_step(A):
#     r = A[1::-1]
#     return r
