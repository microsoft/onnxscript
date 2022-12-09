"""Commonly shared functions for the function library."""
from __future__ import annotations

import onnx.helper

from onnxscript.onnx_opset import opset18 as op


def ones_like(x, onnx_dtype: int):
    shape = op.Shape(x)
    return op.ConstantOfShape(
        shape, value=onnx.helper.make_tensor("one", onnx_dtype, [1], [1])
    )
