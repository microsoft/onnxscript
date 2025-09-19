from __future__ import annotations

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
        kwargs_str = ", " + ", ".join(f"{k}={_arg_to_str(v)}" for k, v in kwargs.items())
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
    # Outer most frame is the first element. This is a reversed of inspect.stack()
    stack: list[inspect.FrameInfo]


class TracingMode(TorchDispatchMode):
    def __init__(self, *args, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces: list[Trace] = []
        self._verbose = verbose

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        stack = reversed(inspect.stack()[1:])  # Exclude the current frame
        # Filter out frames from PyTorch internals
        stack = [frame for frame in stack if "site-packages/torch" not in frame.filename]
        op_str = _op_to_str(func, *args, **kwargs)
        self._add_trace(Trace(op_str, stack))

        result = func(*args, **kwargs)

        return result

    def _add_trace(self, trace: Trace) -> None:
        self.traces.append(trace)
        if self._verbose:
            self._print_last_trace()

    def _print_last_trace(self) -> None:
        print(self._last_trace_str())

    def _last_trace_str(self) -> str:
        if not self.traces:
            return ""

        trace = self.traces[-1]

        common_length = 0

        if len(self.traces) > 1:
            # Find the common prefix between the current stack and the trace stack
            prev_trace = self.traces[-2]
            for f1, f2 in zip(trace.stack, prev_trace.stack):
                if f1.filename == f2.filename and f1.lineno == f2.lineno:
                    common_length += 1
                else:
                    break
            if common_length == len(trace.stack):
                # Keep at least one frame to show the context of the operator
                common_length -= 1
            relevant_stack = trace.stack[common_length:]
        else:
            relevant_stack = trace.stack

        lines = []
        for i, frame in enumerate(relevant_stack):
            indent = i + common_length
            src_line = frame.code_context[0].strip() if frame.code_context else ""

            lines.append(f'{"| " * indent}{src_line}  # {trace.op_str}; {frame.filename}:{frame.lineno} in {frame.function}')

        return "\n".join(lines)

class Model(torch.nn.Module):
    def forward(self, x, y):
        z = torch.add(x, y)
        return torch.relu(z)


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 3)

with TracingMode():
    out = model(x, y)
