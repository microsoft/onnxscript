from __future__ import annotations

import logging
from typing import Sequence

import onnx
from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
)

logger = logging.getLogger(__name__)


def remove_unused_optional_outputs(
    n: onnx.NodeProto, used: set, opset_import: Sequence[onnx.OperatorSetIdProto]
) -> None:
    try:
        if n.domain not in {"", "onnx.ai"}:
            return
        onnx_opset_version = 1
        for opset in opset_import:
            if opset.domain == n.domain:
                onnx_opset_version = opset.version
        op_schema = onnx.defs.get_schema(n.op_type, onnx_opset_version, domain=n.domain)
    except Exception:
        return
    # TODO: If current node is a BatchNormalization node,
    # based on training_mode atrribute, number of optional outputs and
    # how they are handled varies, handle both training_modes
    if n.op_type == "BatchNormalization":
        return
    optional_info = []
    for o in op_schema.outputs:
        # Current ops do not have optional outputs if they have variable number of outputs
        if o.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
            return
        optional_info.append(o.option == onnx.defs.OpSchema.FormalParameterOption.Optional)
    # If no optional outputs in spec, skip delete operations
    if len([o == 1 for o in optional_info]) == 0:
        return

    for i, out in enumerate(n.output):
        if out not in used and optional_info[i] is True:
            n.output[i] = ""
    # Only delete trailing unused optional outputs
    for o in n.output[::-1]:  # type: ignore[assignment]
        if o == "":
            n.output.pop()
        else:
            return


def compute_used_in_node(n: onnx.NodeProto) -> set[str]:
    used = {n for n in n.input if n != ""}
    for attr in n.attribute:
        if attr.HasField("g"):
            used |= compute_used_in_graph(attr.g)
        elif len(attr.graphs) > 0:
            for graph in attr.graphs:
                used |= compute_used_in_graph(graph)
    return used


def compute_used_in_graph(g: onnx.GraphProto) -> set[str]:
    used = set()
    for n in g.node:
        used |= compute_used_in_node(n)
    return used


def process_nodes(
    nodes: RepeatedCompositeFieldContainer[onnx.NodeProto],
    used: set,
    opset_import: Sequence[onnx.OperatorSetIdProto],
) -> int:
    count = 0
    i = len(nodes) - 1
    while i >= 0:
        node = nodes[i]
        remove_unused_optional_outputs(node, used, opset_import)
        used_outputs = [x for x in node.output if x in used]
        if not used_outputs:
            del nodes[i]
            count += 1
            i -= 1
            continue
        for attr in node.attribute:
            if attr.HasField("g"):
                process_graph(attr.g, opset_import)
            elif len(attr.graphs) > 0:
                for graph in attr.graphs:
                    process_graph(graph, opset_import)
        used |= compute_used_in_node(node)
        i -= 1
    return count


def process_graph(
    graph: onnx.GraphProto, opset_import: Sequence[onnx.OperatorSetIdProto]
) -> int:
    used = {output.name for output in graph.output}

    count = process_nodes(graph.node, used, opset_import)

    for i in range(len(graph.initializer) - 1, -1, -1):
        if graph.initializer[i].name not in used:
            del graph.initializer[i]
            count += 1

    return count


def process_function(
    function: onnx.FunctionProto, opset_import: Sequence[onnx.OperatorSetIdProto]
) -> int:
    used = set(function.output)

    return process_nodes(function.node, used, opset_import)


def remove_unused_nodes(model: onnx.ModelProto) -> None:
    """Removes unused nodes from the model."""
    count = process_graph(model.graph, model.opset_import)
    for function in model.functions:
        count += process_function(function, model.opset_import)

    logger.info("Removed %s unused nodes", count)
