# SPDX-License-Identifier: Apache-2.0


def Dropout(data, ratio, training_mode, seed: int):
    if (training_mode):
        mask = onnx.ConstantOfShape(onnx.Shape(data), value=True)
        output = data
    else:
        rand = onnx.RandomUniformLike(data, dtype=1, seed=seed)
        mask = (rand >= ratio)
        output = onnx.Where(mask, data, 0) / (1.0 - ratio)
    return (output, mask)
