"""All operators in torch.ops.aten.

- No inplace operators.
"""
import onnxscript
from onnxscript.onnx_opset import default_opset as op

from beartype import Is
from typing_extensions import Annotated


def atenop(name, spec=None):
    """A no-op decorator for torch.aten operators."""

    def decorator(func):
        return func

    return decorator


# TODO: put this in nn
@atenop("aten::relu6")
def Relu6(X):
    zero = op.CastLike(0, X)
    return op.Max(X, zero)


@atenop("aten::selu")
def Selu(self):
    return op.Selu(self)


@onnxscript.script()
@atenop("aten::elu")
def Elu(
    self,
    alpha: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
    scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
    input_scale: float = 1.0,
):
    return op.Elu(self, alpha=alpha)
