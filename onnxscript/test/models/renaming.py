# SPDX-License-Identifier: Apache-2.0

from onnxscript.types import FLOAT

# same variable assigned multiple times


def renaming(A: FLOAT["N"]) -> FLOAT["N"]:
    T = onnx.Abs(A)
    T = onnx.Neg(A)
    return T

# clash between generated-name and pre-existing name


def renaming2(A: FLOAT["N"]) -> FLOAT["N"]:
    T_0 = onnx.Relu(A)
    T = onnx.Abs(A)
    T = onnx.Neg(A)
    return T
