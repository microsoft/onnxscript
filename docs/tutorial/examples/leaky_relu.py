# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def LeakyRelu(X, alpha: float):
    return op.Where(X < 0.0, alpha * X, X)
