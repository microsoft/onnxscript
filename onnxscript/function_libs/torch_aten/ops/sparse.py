# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `sparse` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from onnxscript import TensorType


def aten_sparse_sampled_addmm(
    self: TensorType, mat1: TensorType, mat2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()
