"""Common operators shared in the torchlib library."""

import onnxscript
import onnxscript.values
from onnxscript import BOOL, INT64
from onnxscript import opset18 as op
from onnxscript.function_libs.torch_lib import _constants, tensor_typing

DOMAIN = f"{_constants.DOMAIN}.common"

common_opset = onnxscript.values.Opset(domain=DOMAIN, version=1)


@onnxscript.script(common_opset)
def Rank(input: tensor_typing.TTensor) -> INT64:
    """Take the rank of the input tensor."""

    return op.Size(op.Shape(input))


@onnxscript.script(common_opset)
def IsScalar(input: tensor_typing.TTensor) -> BOOL:
    """Return whether the input has rank 0, or is a scalar."""

    return op.Equal(op.Size(op.Shape(input)), op.Constant(value_int=0))
