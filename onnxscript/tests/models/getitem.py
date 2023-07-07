# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import INT32, INT64
from onnxscript.tests.common.onnx_script_test_case import FunctionTestParams

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
zero = np.array(0, dtype=np.int64)

# Inputs/Outputs of test-cases are specified as numpy arrays, or tuples of numpy arrays,
# or as a list of values, e.g. [0, 1, 2], converted to an int32 numpy array.

IOType = Union[np.ndarray, Tuple[np.ndarray, ...], list]


def wrap_input_output(x: IOType) -> list(np.ndarray):
    if isinstance(x, np.ndarray):
        return [x]
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [np.array(x, dtype=np.int32)]


def test(f, input: IOType, output: IOType) -> FunctionTestParams:
    return FunctionTestParams(f, wrap_input_output(input), wrap_input_output(output))


@script(default_opset=op)
def get_first_row(A: INT32[...]) -> INT32[...]:
    return A[0]


get_first_row_test = test(get_first_row, input=x, output=[0, 1, 2])


@script(default_opset=op)
def get_last_row(A: INT32[...]) -> INT32[...]:
    return A[-1]


get_last_row_test = test(get_last_row, input=x, output=[9, 10, 11])


@script(default_opset=op)
def get_column(A: INT32[...]) -> INT32[...]:
    return A[:, 1]


get_column_test = test(get_column, input=x, output=[1, 4, 7, 10])


@script(default_opset=op)
def get_unknown_row(A: INT32[...], i: INT64[...]) -> INT32[...]:
    return A[i]


get_unknown_row_test = test(get_unknown_row, input=(x, zero), output=[0, 1, 2])


@script(default_opset=op)
def get_computed_row(A: INT32[...]) -> INT32[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1]


get_computed_row_test = test(get_computed_row, input=x, output=[3, 4, 5])


@script(default_opset=op)
def slice_from_1_to_2(A: INT32[...]) -> INT32[...]:
    return A[1:2]


slice_from_to = test(slice_from_1_to_2, input=x, output=[[3, 4, 5]])


@script(default_opset=op)
def slice_from_1(A: INT32[...]) -> INT32[...]:
    return A[1:]


slice_from = test(slice_from_1, input=x, output=[[3, 4, 5], [6, 7, 8], [9, 10, 11]])


@script(default_opset=op)
def slice_to_2(A: INT32[...]) -> INT32[...]:
    return A[:2]


slice_to = test(slice_to_2, input=x, output=[[0, 1, 2], [3, 4, 5]])


@script(default_opset=op)
def slice_step_minus1(A: INT32[...]) -> INT32[...]:
    return A[::-1]


slice_neg_step = test(
    slice_step_minus1, input=x, output=[[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]]
)


@script(default_opset=op)
def slice_from_1_to_minus1(A: INT32[...]) -> INT32[...]:
    return A[1:-1]


slice_from_neg_to = test(slice_from_1_to_minus1, input=x, output=[[3, 4, 5], [6, 7, 8]])


@script(default_opset=op)
def slice_to_0_step_minus1(A: INT32[...]) -> INT32[...]:
    return A[:0:-1]


slice_to_neg_step = test(slice_to_0_step_minus1, input=x, output=x[:0:-1])


@script(default_opset=op)
def get_row_slice_column(A: INT32[...]) -> INT32[...]:
    return A[0, :0:-1]


index_and_slice = test(get_row_slice_column, input=x, output=x[0, :0:-1])


@script(default_opset=op)
def slice_row_get_column(A: INT32[...]) -> INT32[...]:
    return A[:2, 0]


slice_and_index = test(slice_row_get_column, input=x, output=x[:2, 0])


@script(default_opset=op)
def slice_row_and_column(A: INT32[...]) -> INT32[...]:
    return A[:2, :1]


slice_and_slice = test(slice_row_and_column, input=x, output=x[:2, :1])


@script(default_opset=op)
def slice_from_2_to_0_step_minus1(A: INT32[...]) -> INT32[...]:
    return A[2:0:-1]


slice_from_to_neg_step = test(slice_from_2_to_0_step_minus1, input=x, output=x[2:0:-1])


@script()
def slice_computed_range(A: INT32[...]) -> INT32[...]:
    scalar_zero = op.Constant(value=make_tensor("scalar_zero", TensorProto.INT64, [], [0]))
    return A[scalar_zero + 1 : scalar_zero + 2]


slice_computed_range_test = test(slice_computed_range, input=x, output=x[1:2])


@script(default_opset=op)
def nested_expr(A: INT32[...]) -> INT32[...]:
    r = (A + 1)[0]
    return r


nested_expr_test = test(nested_expr, input=x, output=[1, 2, 3])

# This notation is not possible with ONNX but is allowed by numpy.
# @script()
# def getitem_i_slice_right_step(A):
#     r = A[1::-1]
#     return r
