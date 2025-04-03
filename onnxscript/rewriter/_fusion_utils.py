# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

from onnxscript import ir

Dim = Union[int, ir.SymbolicDim]


def _check_shape(bindings: dict[str, Dim], val: ir.Value, shape: Sequence[str]) -> bool:
    if val.shape is None:
        return False
    if val.shape.rank() != len(shape):
        return False
    for actual, expected in zip(val.shape, shape):
        if expected not in bindings:
            bindings[expected] = actual  # type: ignore[assignment]
        elif actual != bindings[expected]:
            return False
    return True
