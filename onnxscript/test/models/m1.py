# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT


def cadd(A: FLOAT[1, 2], B: FLOAT[1, 2]) -> FLOAT[1, 4]:
    return op.Concat(A, B, axis=-1)
