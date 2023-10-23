"""Common operators shared in the torchlib library."""

import onnxscript
from onnxscript import INT64, BOOL
from onnxscript import opset18 as op
import onnxscript.values
from onnxscript.function_libs.torch_lib import _constants, tensor_typing

DOMAIN = f"{_constants.DOMAIN}.common"


class CommonOpset(onnxscript.values.Opset):
    def __new__(cls):
        return onnxscript.values.Opset.__new__(cls, DOMAIN, 1)

    def Rank(self, x: tensor_typing.TTensor) -> INT64:
        return Rank(x)

    def IsScalar(self, x: tensor_typing.TTensor) -> BOOL:
        return IsScalar(x)


common_opset = CommonOpset()

# TODO(justinchuby): How does onnxscript know what an op is a function?
# Or do we need onnxscript to support calling functions from a different module?

@onnxscript.script(common_opset)
def Rank(input: tensor_typing.TTensor) -> INT64:
    """Take the rank of the input tensor."""

    return op.Size(op.Shape(input))

@onnxscript.script(common_opset)
def IsScalar(input: tensor_typing.TTensor) -> BOOL:
    """Return whether the input has rank 0, or is a scalar."""

    return op.Size(op.Shape(input)) == 0
