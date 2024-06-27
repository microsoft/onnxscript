# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnx import TensorProto, helper

from onnxscript import opset15 as op
from onnxscript import script

script_const = helper.make_tensor("scalar_half", TensorProto.FLOAT, (), [0.5])


@script()
def tensor_attr(x):
    c = op.Constant(value=script_const)
    return c * x


# The following assignment has no effect on the ONNX FunctionProto
# generated from tensor_attr:


script_const = helper.make_tensor("scalar_one", TensorProto.FLOAT, (), [1.0])

fp = tensor_attr.to_function_proto()
