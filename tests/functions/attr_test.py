# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import numpy as np
import onnx
import onnx.helper

from onnxscript import script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.tests.common.onnx_script_test_case import FunctionTestParams as Test


@script()
def int_attr():
    return op.Constant(value_int=17)


int_attr_test = Test(int_attr, input=[], output=[np.array(17, dtype=np.int64)])


@script()
def ints_attr():
    return op.Constant(value_ints=[17, 19])


ints_attr_test = Test(ints_attr, input=[], output=[np.array([17, 19], dtype=np.int64)])


@script()
def float_attr():
    return op.Constant(value_float=17.0)


float_attr_test = Test(float_attr, input=[], output=[np.array(17.0, dtype=np.float32)])


@script()
def floats_attr():
    return op.Constant(value_floats=[17.0, 19.0])


floats_attr_test = Test(
    floats_attr, input=[], output=[np.array([17.0, 19.0], dtype=np.float32)]
)


@script()
def string_attr():
    return op.Constant(value_string="hello")


string_attr_test = Test(string_attr, input=[], output=[np.array("hello")])


@script()
def strings_attr():
    return op.Constant(value_strings=["hello", "world"])


strings_attr_test = Test(strings_attr, input=[], output=[np.array(["hello", "world"])])

tensor1 = onnx.helper.make_tensor("t", onnx.TensorProto.FLOAT, [2], [17.0, 19.0])


@script()
def tensor_attr():
    return op.Constant(value=tensor1)


tensor_attr_test = Test(
    tensor_attr, input=[], output=[np.array([17.0, 19.0], dtype=np.float32)]
)
