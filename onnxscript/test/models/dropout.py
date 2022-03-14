# SPDX-License-Identifier: Apache-2.0


def Dropout(data, ratio, training_mode, seed: int):
    if (training_mode):
        mask = oxs.ConstantOfShape(oxs.Shape(data), value=True)
        output = data
    else:
        rand = oxs.RandomUniformLike(data, dtype=1, seed=seed)
        mask = (rand >= ratio)
        output = oxs.Where(mask, data, 0) / (1.0 - ratio)
    return (output, mask)
