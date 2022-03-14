# SPDX-License-Identifier: Apache-2.0

from onnxscript.onnx_types import FLOAT


def multi(A: FLOAT["N"]) -> FLOAT["N"]:
    x, y = oxs.Foo(A)
    x, y = oxs.Bar(A)
    return x + y
