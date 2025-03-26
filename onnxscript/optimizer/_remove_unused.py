# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx

import onnxscript.optimizer._legacy._remove_unused_proto
from onnxscript import ir

logger = logging.getLogger(__name__)


def remove_unused_optional_outputs(
    node: ir.Node, graph_outputs: frozenset[ir.Value], onnx_opset_version: int
) -> None:
    try:
        if node.domain not in {"", "onnx.ai"}:
            return
        op_schema = onnx.defs.get_schema(node.op_type, onnx_opset_version, domain=node.domain)
    except Exception:
        return

    if node.op_type == "BatchNormalization":
        # BatchNormalization op has 3 outputs: Y, running_mean, running_var
        # If running_mean and running_var are not used, remove them, and the training_mode attribute
        def is_used_output(i: int) -> bool:
            if i < len(node.outputs):
                val = node.outputs[i]
                return val in graph_outputs or bool(val.uses())
            return False

        if is_used_output(1) or is_used_output(2):
            return
        if len(node.outputs) > 1:
            node.outputs[1].name = ""
        if len(node.outputs) > 2:
            node.outputs[2].name = ""
        node.attributes.pop("training_mode", None)
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

    for i, out in enumerate(node.outputs):
        if out not in graph_outputs and (not out.uses()) and optional_info[i] is True:
            out.name = ""


def _process_function_or_graph(function_or_graph: ir.Function | ir.Graph) -> int:
    graph_outputs = frozenset(function_or_graph.outputs)
    onnx_opset_version = function_or_graph.opset_imports.get("", None)
    count = 0
    for node in reversed(function_or_graph):
        removable = True
        for output in node.outputs:
            if output in graph_outputs or output.uses():
                removable = False
                break
        if removable:
            function_or_graph.remove(node, safe=True)
            count += 1
        else:
            if onnx_opset_version is not None:
                remove_unused_optional_outputs(node, graph_outputs, onnx_opset_version)
            for attr in node.attributes.values():
                if not isinstance(attr, ir.Attr):
                    continue
                if attr.type == ir.AttributeType.GRAPH:
                    count += _process_function_or_graph(attr.as_graph())
                elif attr.type == ir.AttributeType.GRAPHS:
                    for graph in attr.as_graphs():
                        count += _process_function_or_graph(graph)
    return count


class RemoveUnusedNodesPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = _process_function_or_graph(model.graph)
        graph_outputs = frozenset(model.graph.outputs)
        initializers = model.graph.initializers
        for init in list(initializers.values()):
            if not (init in graph_outputs or init.uses()):
                assert init.name is not None
                del initializers[init.name]
                count += 1
        for function in model.functions.values():
            count += _process_function_or_graph(function)
        if count:
            logger.info("Removed %s unused nodes", count)
            return ir.passes.PassResult(model, modified=True)
        return ir.passes.PassResult(model, modified=False)


def remove_unused_nodes(model: ir.Model | onnx.ModelProto) -> None:
    """Removes unused nodes from a model."""
    if isinstance(model, ir.Model):
        RemoveUnusedNodesPass()(model)
    else:
        onnxscript.optimizer._legacy._remove_unused_proto.remove_unused_nodes(model)
