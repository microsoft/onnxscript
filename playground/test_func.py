from onnxscript import script
from onnxscript.onnx_opset import opset17 as op
from onnxscript import onnx_types as ot

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, AbstractSet, Tuple

import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo.core import OpInfo

TESTED_DTYPES = (
    # torch.bool,
    # torch.uint8,
    # torch.int8,
    # torch.int16,
    # torch.int32,
    # torch.int64,
    # Floating types
    # torch.float16,
    torch.float32,
    torch.float64,
    # torch.bfloat16,
    # Complex types
    # torch.complex32,
    # torch.complex64,
    # torch.complex128,
)

# from pyinstrument import Profiler

# profiler = Profiler()
# profiler.start()

@script()
def LeakyRelu(input, negative_slope: float = 0.01, inplace: bool = False):
    zero = op.CastLike(0, input)
    negative_slope = op.CastLike(negative_slope, input)
    return op.Where(input < zero, negative_slope * input, input)


def produce_op_sample(
    skip_ops: AbstractSet[str] | None = None, target: str | None = None
) -> None:
    skip_ops = skip_ops or set()

    # opinfo_definitions.op_db is part of common_methods_invocations.op_db
    op_db = common_methods_invocations.op_db
    for op_info in op_db:
        if op_info.name not in {"nn.functional.leaky_relu"}:
            continue
        for dtype in TESTED_DTYPES:
            for i, sample in enumerate(
                op_info.sample_inputs(
                    device="cpu", dtype=dtype, requires_grad=False
                )
            ):
                print(i, ": ", sample)

                inputs = (
                    sample.input,
                    *sample.args,
                )
                inputs_np = [np.array(input) if isinstance(input, torch.Tensor) else input for input in inputs]
                torch_out = op_info.op(*inputs, **sample.kwargs)
                onnx_out = LeakyRelu(*inputs_np, **sample.kwargs)

                print("torch_out: ", torch_out)
                print("onnx_out: ", onnx_out)

def main():

    result = LeakyRelu(np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32), negative_slope=0.1)
    print(result)
    print(type(result))
    produce_op_sample()


# profiler.stop()

# profiler.print()

if __name__ == "__main__":
    main()