# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript.main import script
from onnxscript.onnx_opset import opset14, opset16
from onnxscript.onnx_types import INT64


@script()
def shape_A(data, start: INT64[1], end: INT64[1]):
    shape = opset16.Shape(data)
    zero = opset16.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if end == zero:
        length = opset16.Shape(opset16.Shape(data))
        res = opset16.Slice(shape, start, length, zero)
    else:
        res = opset16.Slice(shape, start, end, zero)
    return res


@script()
def shape_B(data, start: INT64[1], end: INT64[1]):
    shape = opset14.Shape(data)
    zero = opset14.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    if end == zero:
        length = opset14.Shape(opset14.Shape(data))
        res = opset14.Slice(shape, start, length, zero)
    else:
        res = opset14.Slice(shape, start, end, zero)
    return res


@script(default_opset=opset16)
def inc_any(data):
    # The converter cannot know which opset to use unless it is specified
    # in the decorator.
    return data + 1
