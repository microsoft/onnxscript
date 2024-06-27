# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def LeakyRelu(X, alpha: float):
    alpha_value = op.Constant(value_float=alpha)
    return op.Where(X < 0.0, alpha_value * X, X)
