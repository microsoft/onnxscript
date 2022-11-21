# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from onnx.defs import onnx_opset_version

from onnxscript import values

if onnx_opset_version() < 14:
    raise ImportError(
        f"onnx-script requires onnx opset >= 14 but {onnx_opset_version()} is detected."
    )

opset1 = values.Opset("", 1)
opset2 = values.Opset("", 2)
opset3 = values.Opset("", 3)
opset4 = values.Opset("", 4)
opset5 = values.Opset("", 5)
opset6 = values.Opset("", 6)
opset7 = values.Opset("", 7)
opset8 = values.Opset("", 8)
opset9 = values.Opset("", 9)
opset10 = values.Opset("", 10)
opset11 = values.Opset("", 11)
opset12 = values.Opset("", 12)
opset13 = values.Opset("", 13)
opset14 = values.Opset("", 14)

if onnx_opset_version() >= 15:
    opset15 = values.Opset("", 15)

if onnx_opset_version() >= 16:
    opset16 = values.Opset("", 16)

if onnx_opset_version() >= 17:
    opset17 = values.Opset("", 17)

if onnx_opset_version() >= 18:
    opset18 = values.Opset("", 18)

default_opset = values.Opset("", onnx_opset_version())

onnxml1 = values.Opset("ai.onnx.ml", 1)
onnxml2 = values.Opset("ai.onnx.ml", 2)
onnxml3 = values.Opset("ai.onnx.ml", 3)
