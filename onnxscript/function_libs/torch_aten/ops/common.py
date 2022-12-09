"""Commonly shared functions for the function library."""
from __future__ import annotations

import onnxscript
from onnxscript.onnx_opset import opset18 as op


@onnxscript.script()
def ones_like(x, dtype: int):
    shape = op.Shape(x)
    one_dtype = op.Cast(1, to=dtype)
    return op.Expand(one_dtype, shape)
