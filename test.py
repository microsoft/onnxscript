import dataclasses
import traceback
import inspect

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._pytree import tree_map


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _tensor_debug_string(tensor) -> str:
    """Convert tensor to debug string representation."""
    if isinstance(tensor, torch.Tensor):
        return f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg) -> str:
    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)



def _op_to_str(op, *args, **kwargs) -> str:
    args_str = ", ".join(_arg_to_str(arg) for arg in args)

    if kwargs:
        kwargs_str = ", " + ", ".join(
            f"{k}={_arg_to_str(v)}" for k, v in kwargs.items()
        )
    else:
        kwargs_str = ""

    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)

    return f"{op_name}({args_str}{kwargs_str})"


@dataclasses.dataclass
class Trace:
    op_str: str
    stack: str



class TracingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = func(*args, **kwargs)

        return result


class Model(torch.nn.Module):
    def forward(self, x, y):
        z = torch.add(x, y)
        return torch.relu(z)


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 3)

with TracingMode():
    out = model(x, y)
