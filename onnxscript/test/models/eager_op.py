# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15 as op


@script()
def eager_op(X: FLOAT[...]) -> FLOAT[...]:
    return X % 1.5


@script()
def eager_abs(X: FLOAT[...]) -> FLOAT[...]:
    return op.Abs(X) + 1.0
