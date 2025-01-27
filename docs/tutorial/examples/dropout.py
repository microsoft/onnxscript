# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript import opset15 as op
from onnxscript import script


@script()
def Dropout(data, ratio, training_mode, seed: float):
    if training_mode:
        rand = op.RandomUniformLike(data, dtype=1, seed=seed)
        mask = rand >= ratio
        output = op.Where(mask, data, 0) / (1.0 - ratio)
    else:
        mask = op.ConstantOfShape(op.Shape(data), value=True)
        output = data
    return (output, mask)
