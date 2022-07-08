# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx.defs import onnx_opset_version
from .values import Opset


if onnx_opset_version() < 14:
    raise ImportError(
        f"onnx-script requires onnx opset >= 14 but {onnx_opset_version()} is detected.")

opset1 = Opset("", 1)
opset2 = Opset("", 2)
opset3 = Opset("", 3)
opset4 = Opset("", 4)
opset5 = Opset("", 5)
opset6 = Opset("", 6)
opset7 = Opset("", 7)
opset8 = Opset("", 8)
opset9 = Opset("", 9)
opset10 = Opset("", 10)
opset11 = Opset("", 11)
opset12 = Opset("", 12)
opset13 = Opset("", 13)
opset14 = Opset("", 14)

if onnx_opset_version() >= 15:
    opset15 = Opset("", 15)

if onnx_opset_version() >= 16:
    opset16 = Opset("", 16)

if onnx_opset_version() >= 17:
    opset17 = Opset("", 17)
