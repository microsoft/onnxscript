# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

import onnxscript.ir as ir
from onnxscript.optimizer import basic_constant_propagation


def display_nodes(nodes: Sequence[ir.Node]) -> None:
    """Display a list of nodes in the order they appear in the graph."""
    if nodes:
        graph = nodes[0].graph
        if graph:
            # Display nodes in same order as in graph:
            # Currently doesn't handle (control-flow) subgraphs
            for node in graph:
                if node in nodes:
                    node.display()
        else:
            for node in nodes:
                node.display()


def display_slice(x: ir.Value | ir.Node, backward: bool = True, depth_limit: int = 5) -> None:
    """Display the (backward or forward) subgraph from a given value or node upto a certain depth."""
    slice = []

    def visit(node: ir.Node, depth):
        if node in slice:
            return
        slice.append(node)
        if depth < depth_limit:
            if backward:
                for inp in node.inputs:
                    if inp is not None and inp.producer() is not None:
                        visit(inp.producer(), depth + 1)  # type: ignore[arg-type]
            else:
                for out in node.outputs:
                    for consumer, _ in out.uses():
                        visit(consumer, depth + 1)

    if isinstance(x, ir.Node):
        visit(x, 0)
    elif isinstance(x, ir.Value) and x.producer() is not None:
        visit(x.producer(), 0)  # type: ignore[arg-type]
    display_nodes(slice)


def get_const_value(value: ir.Value) -> ir.TensorProtocol | None:
    node = value.producer()
    if node is not None:
        basic_constant_propagation([node])
    return value.const_value


def get_numpy_value(val: ir.Value | None) -> np.ndarray | None:
    """Convenience wrapper to get (optional) numpy value from an optional IR Value.

    This is intended for use in optimizations/rewriting. Note that this does not
    yet handle the distinction between inputs with default values (values that are
    both graph inputs and graph initializers), which should not be treated as a
    constant, and true constant values. The caller should make the distinction, as
    a value does not contain enough information to determine this. (TODO)
    """
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


def is_singleton_value(
    val: ir.Value | None, expected: float | int | Callable, *, rtol: float | None = None
) -> bool:
    """Returns True if the value is a single element tensor with given value, and False otherwise."""
    scalar = get_singleton_value(val)
    if scalar is None:
        return False
    if callable(expected):
        return expected(scalar)
    if isinstance(expected, int):
        return expected == scalar
    # rtol must be specified for float comparison
    assert rtol is not None
    return math.isclose(scalar, expected, rel_tol=rtol)


def has_rank(value: ir.Value | None, rank: int) -> bool:
    """Returns True if the value is statically known to have the given rank, and False otherwise."""
    if value is None:
        return False
    shape = value.shape
    return (shape is not None) and (shape.rank() == rank)
