# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Mapping, Sequence

import onnx_ir as ir
from onnx_ir._cloner import Cloner


def instantiate(
    graph: ir.Graph,
    inputs: Sequence[ir.Value | None],
    attributes: Mapping[str, ir.Attr],
    *,
    prefix: str = "",
) -> tuple[list[ir.Node], list[ir.Value | None]]:
    """Instantiate (inline) a graph, substituting inputs and attributes.

    Args:
        graph: The graph to instantiate.
        inputs: Actual input values to bind to the graph's formal parameters.
        attributes: Attribute values to substitute for reference attributes.
        prefix: Optional prefix to prepend to node and output names.

    Returns:
        A tuple of (nodes, outputs) where nodes are the cloned graph body
        and outputs are the values corresponding to the graph's outputs.
    """
    formal_inputs = graph.inputs
    if len(inputs) > len(formal_inputs):
        raise ValueError(
            f"Too many inputs: got {len(inputs)}, "
            f"but graph has {len(formal_inputs)} parameters."
        )
    value_map: dict[ir.Value, ir.Value | None] = dict(zip(formal_inputs, inputs))

    def rename(node: ir.Node) -> None:
        if prefix:
            if node.name:
                node.name = prefix + node.name
            for output in node.outputs:
                if output is not None and output.name:
                    output.name = prefix + output.name

    cloner = Cloner(
        attr_map=attributes,
        value_map=value_map,
        metadata_props={},
        post_process=rename,
        resolve_ref_attrs=True,
        allow_outer_scope_values=True,
    )
    nodes = [cloner.clone_node(n) for n in graph]
    outputs = [value_map.get(v) for v in graph.outputs]
    return nodes, outputs
