# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def omitted_input(x):
    # The following two statements are equivalent:
    y1 = op.Clip(x)
    y2 = op.Clip(x, None, None)
    # The following example shows an omitted optional input, followed by another input
    y3 = op.Clip(x, None, 1.0)
    return y1 + y2 + y3
