# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    "RemoveUnusedNodesPass",
    "RemoveUnusedFunctionsPass",
    "RemoveUnusedOpsetsPass",
]

import logging

import onnx

from onnxscript import ir

logger = logging.getLogger(__name__)


def _remove_unused_optional_outputs(
    node: ir.Node, graph_outputs: frozenset[ir.Value], onnx_opset_version: int
) -> None:
    try:
        if node.domain not in {"", "onnx.ai"}:
            return
        op_schema = onnx.defs.get_schema(node.op_type, onnx_opset_version, domain=node.domain)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.info(
            "Failed to get schema for %s, skipping optional output removal",
            node,
            stack_info=True,
        )
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


def _remove_unused_nodes_in_graph_like(function_or_graph: ir.Function | ir.Graph) -> int:
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
                _remove_unused_optional_outputs(node, graph_outputs, onnx_opset_version)
            for attr in node.attributes.values():
                if not isinstance(attr, ir.Attr):
                    continue
                if attr.type == ir.AttributeType.GRAPH:
                    count += _remove_unused_nodes_in_graph_like(attr.as_graph())
                elif attr.type == ir.AttributeType.GRAPHS:
                    for graph in attr.as_graphs():
                        count += _remove_unused_nodes_in_graph_like(graph)
    return count


class RemoveUnusedNodesPass(ir.passes.InPlacePass):
    """Pass for removing unused nodes and initializers (dead code elimination).

    This pass does not modify the model signature (inputs and outputs). It ensures
    that unused nodes and initializers are removed while preserving the original
    contract of the model.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = _remove_unused_nodes_in_graph_like(model.graph)
        graph_outputs = frozenset(model.graph.outputs)
        graph_inputs = frozenset(model.graph.inputs)
        initializers = model.graph.initializers
        for init in list(initializers.values()):
            if not (init.uses() or init in graph_outputs or init in graph_inputs):
                assert init.name is not None
                del initializers[init.name]
                count += 1
        for function in model.functions.values():
            count += _remove_unused_nodes_in_graph_like(function)
        if count:
            logger.info("Removed %s unused nodes", count)
        return ir.passes.PassResult(model, modified=bool(count))


class RemoveUnusedFunctionsPass(ir.passes.InPlacePass):
    def __init__(self):
        super().__init__()
        self._used: set[ir.OperatorIdentifier] | None = None

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        self._used = set()
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            self._call_node(model, node)

        # Update the model to remove unused functions
        unused = set(model.functions) - self._used
        if not unused:
            logger.info("No unused functions to remove")
            return ir.passes.PassResult(model, modified=False)

        for op_identifier in unused:
            del model.functions[op_identifier]

        logger.info("Removed %s unused functions", len(unused))
        logger.debug("Functions left: %s", list(model.functions))
        logger.debug("Functions removed: %s", unused)

        self._used = None
        return ir.passes.PassResult(model, modified=bool(unused))

    def _call_function(self, model: ir.Model, function: ir.Function) -> None:
        assert self._used is not None
        if function.identifier() in self._used:
            # The function and its nodes are already recorded as used
            return
        self._used.add(function.identifier())
        for node in ir.traversal.RecursiveGraphIterator(function):
            self._call_node(model, node)

    def _call_node(self, model: ir.Model, node: ir.Node) -> None:
        op_identifier = node.op_identifier()
        if op_identifier not in model.functions:
            return
        self._call_function(model, model.functions[op_identifier])


class RemoveUnusedOpsetsPass(ir.passes.InPlacePass):
    """Remove unused opset imports from the model and functions.

    Attributes:
        process_functions: Whether to process functions in the model. If True, the pass will
            remove unused opset imports from functions as well. If False, only the main graph
            will be processed.
    """

    def __init__(self, process_functions: bool = True):
        super().__init__()
        self.process_functions = process_functions

    def _process_graph_like(
        self, graph_like: ir.Graph | ir.Function, used_domains: set[str]
    ) -> bool:
        for node in ir.traversal.RecursiveGraphIterator(graph_like):
            used_domains.add(node.domain)
        unused = set(graph_like.opset_imports) - used_domains
        for domain in unused:
            del graph_like.opset_imports[domain]
        return bool(unused)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Record domains of all functions
        used_domains = {""}  # By default always retain the onnx (default) domain
        for function in model.functions.values():
            used_domains.add(function.domain)
        modified = self._process_graph_like(model.graph, used_domains=used_domains)

        if self.process_functions:
            for function in model.functions.values():
                modified |= self._process_graph_like(function, used_domains={""})

        return ir.passes.PassResult(model, modified=modified)
