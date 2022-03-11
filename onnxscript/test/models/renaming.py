# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT

# same variable assigned multiple times


def renaming(A: FLOAT["N"]) -> FLOAT["N"]:
    T = oxs.Abs(A)
    T = oxs.Neg(A)
    return T

# clash between generated-name and pre-existing name


def renaming2(A: FLOAT["N"]) -> FLOAT["N"]:
    T_0 = oxs.Relu(A)
    T = oxs.Abs(A)
    T = oxs.Neg(A)
    return T
