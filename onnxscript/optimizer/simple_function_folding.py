"""Inlines the function if it only contains very few number of nodes."""

from __future__ import annotations

import logging
from typing import Sequence

import onnx

import onnxscript._legacy_ir as ir
from onnxscript._legacy_ir import visitor
from onnxscript.optimizer import remove_unused

logger = logging.getLogger(__name__)


class FunctionInliner(visitor.FunctionCallsiteProtoTransformer):
    counts: dict[ir.FunctionId, int]

    def __init__(self, node_count: int) -> None:
        super().__init__()
        self._node_count = node_count

    def _gather_function_metadata(self, model: onnx.ModelProto) -> None:
        super()._gather_function_metadata(model)
        self._function_renamer._postfix = "inlined"

    def visit_model(self, model: onnx.ModelProto) -> None:
        self.counts = {}

        super().visit_model(model)

    def should_inline_function(self, function: onnx.FunctionProto) -> bool:
        return len(function.node) <= self._node_count

    def process_function_node(
        self, node: onnx.NodeProto
    ) -> tuple[list[onnx.NodeProto] | None, onnx.FunctionProto | None]:
        # Recursively process sub nodes first.
        function_id = (node.domain, node.op_type, getattr(node, "overload", ""))
        function = self._functions[function_id]
        replacement, new_function = super().process_function_node(node)
        function = new_function if new_function else function

        if self.should_inline_function(function):
            self.enter_function_scope(function)
            sub_scope = self.exit_function_scope(function)
            new_nodes = []

            formal_outs = function.output
            actual_outs = node.output
            formal_ins = function.input
            actual_ins = node.input
            # TODO: Potential collision when actual is "".
            # formal.name may collide with existing value names.
            input_renamings = dict(zip(formal_ins, actual_ins))
            if len(actual_ins) < len(formal_ins):
                input_renamings.update(dict.fromkeys(formal_ins[len(actual_ins) :], ""))
            output_renamings = {
                formal: actual
                for formal, actual in zip(formal_outs, actual_outs)
                if actual != ""
            }
            renamings = {**input_renamings, **output_renamings}

            logger.debug("renamings function %s: %s", function.name, renamings)

            def rename(name: str) -> str:
                if name == "":
                    return name
                new_name = renamings.get(name)
                if new_name is None:
                    new_name = f"{node.name}_{name}"
                logger.debug("renaming %s to %s", name, new_name)
                if (ir_value := sub_scope.lookup(name)) is not None:
                    if ir_value.tensor_shape_proto() is not None and ir_value.type is not None:
                        ir_value.name = new_name
                        self.bind(new_name, ir_value)
                return new_name

            ref_attrs = {attr.name: attr for attr in node.attribute}
            # logger.debug("inlining simple function %s. Ref attrs: %s", function.name, ref_attrs)

            def fill_in_ref(attr: onnx.AttributeProto) -> onnx.AttributeProto:
                if attr.ref_attr_name:
                    new_attr = onnx.AttributeProto()
                    new_attr.CopyFrom(ref_attrs[attr.ref_attr_name])
                    new_attr.name = attr.name
                    return new_attr
                return attr

            def update_graph_attribute(
                attr: onnx.AttributeProto,
            ) -> onnx.AttributeProto:
                if attr.g:
                    new_attr = onnx.AttributeProto()
                    new_attr.CopyFrom(attr)
                    for node in new_attr.g.node:
                        node.input[:] = [rename(name) for name in node.input]
                        node.output[:] = [rename(name) for name in node.output]
                        new_attrs = []
                        for attr in node.attribute:
                            new_attrs.append(update_attribute(attr))
                        del node.attribute[:]
                        node.attribute.extend(new_attrs)
                    for vi_proto in new_attr.g.input:
                        vi_proto.name = rename(vi_proto.name)
                    for vi_proto in new_attr.g.output:
                        vi_proto.name = rename(vi_proto.name)
                    return new_attr
                return attr

            def update_attribute(attr: onnx.AttributeProto) -> onnx.AttributeProto:
                new_attr = fill_in_ref(attr)
                new_attr = update_graph_attribute(new_attr)
                return new_attr

            for sub_node in function.node:
                # logger.debug("inlining simple function. old node: %s", sub_node)
                new_node = onnx.NodeProto()
                new_node.CopyFrom(sub_node)
                new_node.input[:] = [rename(name) for name in new_node.input]
                new_node.output[:] = [rename(name) for name in new_node.output]
                del new_node.attribute[:]
                for attr in sub_node.attribute:
                    new_node.attribute.append(update_attribute(attr))
                # Avoid name collision.
                new_node.name = f"{node.name}_{new_node.name}"
                # logger.debug("inlining simple function. new node: %s", new_node)
                new_nodes.append(new_node)

            self.counts.setdefault(function_id, 0)
            self.counts[function_id] += 1

            return new_nodes, None

        return replacement, new_function


class SelectedFunctionInliner(FunctionInliner):
    def __init__(self, functions_to_inline: Sequence[onnx.FunctionProto]):
        super().__init__(node_count=0)  # node_count unused.
        self._functions_to_inline = functions_to_inline

    def should_inline_function(self, function: onnx.FunctionProto) -> bool:
        return function in self._functions_to_inline


class FindFunctionWithUnusedOutputsVisitor(visitor.ProtoVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._function_with_unused_outputs: dict[ir.FunctionId, onnx.FunctionProto] = {}
        self._functions: dict[ir.FunctionId, onnx.FunctionProto] = {}
        self._used_nodes: list[onnx.NodeProto] = []

    def _find_nodes_with_any_unused_output(
        self, nodes: Sequence[onnx.NodeProto], used_values: set[str]
    ) -> list[onnx.NodeProto]:
        target_nodes = []
        for i in range(len(nodes) - 1, -1, -1):
            node = nodes[i]
            if any(x not in used_values for x in node.output):
                # Any unused output means the node is a target node.
                target_nodes.append(node)
            if all(x not in used_values for x in node.output):
                # All unused output means the node is not used at all.
                # Hence do not update used_values with the node's inputs.
                continue
            used_values |= remove_unused.compute_used_in_node(node)
        return target_nodes

    def visit_model(self, model: onnx.ModelProto) -> None:
        used_values = {output.name for output in model.graph.output}
        target_nodes = self._find_nodes_with_any_unused_output(model.graph.node, used_values)

        for function in model.functions:
            self._functions[
                (function.domain, function.name, getattr(function, "overload", ""))
            ] = function
            used_values = set(function.output)
            target_nodes.extend(
                self._find_nodes_with_any_unused_output(function.node, used_values)
            )

        for node in target_nodes:
            if visitor.is_local_function_node(node, self._functions):
                function_id = (node.domain, node.op_type, getattr(node, "overload", ""))
                self._function_with_unused_outputs[function_id] = self._functions[function_id]

        logger.info(
            "Found %s function nodes that have unused outputs.",
            len(self._function_with_unused_outputs),
        )
        for key in self._function_with_unused_outputs:
            logger.info("Function node with unused outputs: %s::%s", key[0], key[1])

    @property
    def function_with_unused_outputs(self) -> dict[ir.FunctionId, onnx.FunctionProto]:
        return self._function_with_unused_outputs


def inline_simple_functions(model: onnx.ModelProto, node_count: int = 2) -> bool:
    """Inlines simple functions based on a node count threshold"""
    inliner = FunctionInliner(node_count)
    inliner.visit_model(model)
    logger.info(
        "inlined %s simple functions based on node count threshold %s.",
        len(inliner.counts),
        node_count,
    )
    for op in inliner.counts:
        logger.info(
            "Inlined simple function '%s::%s' %s times.",
            op[0],
            op[1],
            inliner.counts[op],
        )
    return inliner.modified


def inline_functions_with_unused_outputs(model: onnx.ModelProto) -> bool:
    """Inlines function nodes that have unused outputs."""
    # TODO: Use onnx.inliner after 1.16.
    # This visitor based inliner is used to ensure the function inner value info remains consistent.
    visitor = FindFunctionWithUnusedOutputsVisitor()
    visitor.visit_model(model)
    # FIXME: Fix the type of the argument passed into SelectedFunctionInliner
    inliner = SelectedFunctionInliner(visitor.function_with_unused_outputs.values())  # type: ignore[arg-type]
    inliner.visit_model(model)
    logger.info(
        "inlined %s function nodes that have unused outputs.",
        len(inliner.counts),
    )
    for op in inliner.counts:
        logger.info(
            "Inlined function '%s::%s' %s times.",
            op[0],
            op[1],
            inliner.counts[op],
        )
    return inliner.modified
