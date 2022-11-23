"""All operators in torch.ops.aten.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from beartype.vale import Is
from typing_extensions import Annotated

from onnxscript.onnx_opset import default_opset as op


def atenop(name):
    """A no-op decorator for torch.aten operators."""

    del name

    def decorator(func):
        return func

    return decorator


# TODO: put this in nn
@atenop("aten::relu6")
def Relu6(self):
    zero = op.CastLike(0, self)
    return op.Max(self, zero)


@atenop("aten::selu")
def Selu(self):
    return op.Selu(self)


@atenop("aten::elu")
def Elu(
    self,
    alpha: float = 1.0,
    scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
    input_scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
):
    # del scale
    # del input_scale
    return op.Elu(self, alpha=alpha)
