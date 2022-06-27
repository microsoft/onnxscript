# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx.defs import onnx_opset_version
from .values import Opset


if onnx_opset_version() < 15:
    raise ImportError(
        f"onnx-script requires onnx opset >= 15 but {onnx_opset_version()} is detected.")

opset15 = Opset("", 15)

if onnx_opset_version() >= 16:
    opset16 = Opset("", 16)

if onnx_opset_version() >= 17:
    opset17 = Opset("", 17)
    