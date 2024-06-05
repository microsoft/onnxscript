# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# We use ONNX opset 15 to define the function below.
from onnxscript import opset15 as op
from onnxscript import script


# We use the script decorator to indicate that the following function is meant
# to be translated to ONNX.
@script()
def Softplus(X):
    return op.Log(op.Exp(X) + 1.0)
