# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def tensor_attr(x):
    return op.Mul(0.5, x)
