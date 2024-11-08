# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
from onnxscript.optimizer import basic_constant_propagation


def get_const_value(value: ir.Value) -> ir.TensorProtocol | None:
    node = value.producer()
    if node is not None:
        basic_constant_propagation([node])
    return value.const_value


def get_numpy_value(val: ir.Value | None) -> np.ndarray | None:
    if val is None:
        return None
    const_value = val.const_value
    if const_value is not None:
        try:
            return const_value.numpy()
        except FileNotFoundError:
            # External data is not available.
            return None
    return None


def get_singleton_value(val: ir.Value | None):
    """Returns element of a single element tensor constant value, and None otherwise."""
    np_val = get_numpy_value(val)
    if np_val is not None and np_val.size == 1:
        return np_val.item()
    return None
