# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnx import TensorProto, helper

from onnxscript import opset15 as op
from onnxscript import script


@script()
def tensor_attr(x):
    c = op.Constant(value=helper.make_tensor("scalar_half", TensorProto.FLOAT, (), [0.5]))
    return op.Mul(c, x)
