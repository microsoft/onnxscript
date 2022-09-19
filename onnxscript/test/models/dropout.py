# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript.onnx_opset import opset15 as op


def Dropout(data, ratio, training_mode, seed: int):
    if training_mode:
        rand = op.RandomUniformLike(data, dtype=1, seed=seed)
        mask = rand >= ratio
        output = op.Where(mask, data, 0) / (1.0 - ratio)
    else:
        mask = op.ConstantOfShape(op.Shape(data), value=True)
        output = data
    return (output, mask)
