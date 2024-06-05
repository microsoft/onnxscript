# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def FirstDim(X):
    return op.Shape(X, start=0, end=1)
