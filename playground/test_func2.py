
from onnxscript import script
from onnxscript.onnx_opset import opset17 as op

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, AbstractSet, Tuple


@script()
def LeakyRelu(X, alpha: float):
    return op.Where(X < 0.0, alpha * X, X)


result = LeakyRelu(np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32), alpha=0.01)
print(result)
