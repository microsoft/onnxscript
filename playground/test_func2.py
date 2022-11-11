
from onnxscript import script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT, INT64

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, AbstractSet, Tuple


@script()
def LeakyRelu(X, alpha: float):
    return op.Where(X < 0.0, alpha * X, X)


result = LeakyRelu(np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32), alpha=0.01)
print(result)

@script()
def sumprod(x: FLOAT["N"], M: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
    sum = op.Identity(x)
    prod = op.Identity(x)
    for _ in range(M):
        print(type(M))
        sum = sum + x
        prod = prod * x
    return sum, prod

# print(sumprod.to_function_proto())
print(sumprod(np.array([1, 2, 3], dtype=np.float32), M=2))
