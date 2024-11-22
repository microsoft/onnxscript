# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
from onnxscript.optimizer import basic_constant_propagation


def display_slice(x: ir.Value | ir.Node, backward: bool = True, depth_limit: int = 5):
    """Display the subgraph computing a given value or node upto a certain depth."""
    slice = []
    def visit(node: ir.Node, depth):
        if node in slice:
            return
        slice.append(node)
        if (depth < depth_limit):
            if backward:
                for inp in node.inputs:
                    if inp is not None and inp.producer() is not None:
                        visit(inp.producer(), depth + 1)
            else:
                for out in node.outputs:
                    for consumer, _ in out.uses():
                        visit(consumer, depth + 1)
    if isinstance(x, ir.Node):
        visit(x, 0)
    elif isinstance(x, ir.Value) and x.producer() is not None:
        visit(x.producer(), 0)
    for node in reversed(slice):
        node.display()

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
