# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for building ONNX Scan subgraphs.

Provides helpers for constructing Scan body graphs and compacting padded
Scan outputs.  Used by vision encoders to iterate over variable-size
per-image computations (e.g. rotary position IDs, window indices).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence


def create_body_graph(
    state_inputs: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    name: str = "scan_body",
) -> tuple[ir.Graph, builder.GraphBuilder]:
    """Create a Scan body graph and its builder.

    Args:
        state_inputs: Carry state input values (with names, shapes, types).
        scan_inputs: Per-iteration scan input values.
        name: Name for the body graph.

    Returns:
        ``(body_graph, body_builder)`` — the graph and a builder for
        adding ops to it.
    """
    all_inputs = list(state_inputs) + list(scan_inputs)
    body_graph = ir.Graph(
        all_inputs,
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION},
    )
    body_builder = builder.GraphBuilder(body_graph)
    return body_graph, body_builder


def rename_subgraph_values(graph: ir.Graph, prefix: str) -> None:
    """Rename internal value and node names to avoid collisions with parent.

    ONNX Scan/Loop body graphs share a namespace with the parent graph
    in some runtimes (e.g. ORT).  This function prefixes all node names
    and intermediate value names so they don't collide with the main graph.

    Graph input/output names are NOT renamed — they define the Scan
    interface and must match the Scan op's expectations.
    """
    output_names = {v.name for v in graph.outputs}
    input_names = {v.name for v in graph.inputs}

    for node in graph:
        node.name = prefix + node.name
        for v in node.outputs:
            if v.name not in output_names and v.name not in input_names:
                v.name = prefix + v.name


def compact_scan_output(
    op,
    scan_result,
    lengths_per_iter,
    sentinel: int = -1,
):
    """Remove padding from a Scan output.

    Given a Scan output of shape ``(num_iters, max_len, ...)``, creates a
    boolean mask using ``lengths_per_iter`` to identify valid entries, then
    flattens and compresses to produce a 1D tensor of valid entries.

    Args:
        op: OpBuilder for the main graph.
        scan_result: Scan output ``(num_iters, max_len, ...)``.
        lengths_per_iter: ``(num_iters,)`` INT64 — actual length per iteration.
        sentinel: Value used for padding (default: -1).

    Returns:
        Compacted tensor with padding removed.
    """
    # Flatten first two dims: (num_iters * max_len, ...)
    result_shape = op.Shape(scan_result)
    max_len = op.Gather(result_shape, op.Constant(value_int=1))
    # Build mask: range(max_len) < lengths_per_iter.unsqueeze(1)
    indices = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(max_len),
        op.Constant(value_int=1),
    )  # (max_len,)
    mask_2d = op.Less(
        op.Unsqueeze(indices, [0]),  # (1, max_len)
        op.Unsqueeze(lengths_per_iter, [1]),  # (num_iters, 1)
    )  # (num_iters, max_len)

    flat_mask = op.Reshape(mask_2d, [-1])

    # Handle multi-dim scan outputs: (N*M, D1, D2, ...) -> compact
    ndim = op.Shape(result_shape)  # () — number of dims
    # Flatten first two dims, keep the rest
    tail_shape = op.Slice(result_shape, [2], ndim, [0])
    flat_shape = op.Concat(
        op.Constant(value_ints=[-1]),
        tail_shape,
        axis=0,
    )
    flat_result = op.Reshape(scan_result, flat_shape)

    return op.Compress(flat_result, flat_mask, axis=0)
