# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Mapping, Sequence

import onnx_ir as ir
from onnx_ir._cloner import Cloner


def instantiate(
    function: ir.Function,
    inputs: Sequence[ir.Value | None],
    attributes: Mapping[str, ir.Attr],
    *,
    prefix: str = "",
) -> tuple[list[ir.Node], list[ir.Value | None]]:
    """Instantiate (inline) a function, substituting inputs and attributes.

    Args:
        function: The function to instantiate.
        inputs: Actual input values to bind to the function's formal parameters.
        attributes: Attribute values to substitute for reference attributes.
        prefix: Optional prefix to prepend to node and output names.

    Returns:
        A tuple of (nodes, outputs) where nodes are the cloned function body
        and outputs are the values corresponding to the function's outputs.
    """
    formal_inputs = function.inputs
    if len(inputs) > len(formal_inputs):
        raise ValueError(
            f"Too many inputs: got {len(inputs)}, "
            f"but function has {len(formal_inputs)} parameters."
        )
    value_map: dict[ir.Value, ir.Value | None] = {
        formal: actual for formal, actual in zip(formal_inputs, inputs)
    }

    def rename(node: ir.Node) -> None:
        if prefix:
            node.name = prefix + (node.name or "")
            for output in node.outputs:
                if output is not None:
                    output.name = prefix + (output.name or "")

    cloner = Cloner(
        attr_map=attributes,
        value_map=value_map,
        metadata_props={},
        post_process=rename,
        resolve_ref_attrs=True,
    )
    nodes = [cloner.clone_node(n) for n in function]
    outputs = [value_map.get(v) for v in function.outputs]
    return nodes, outputs

