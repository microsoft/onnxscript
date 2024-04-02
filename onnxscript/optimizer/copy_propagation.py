from __future__ import annotations

from typing import Any

import onnx

import onnxscript.optimizer.remove_unused
from onnxscript._legacy_ir import visitor
from onnxscript.utils.utils import is_onnx_op


class CopyPropagator(visitor.ProtoVisitor):
    def __init__(self):
        super().__init__()

    def visit_node(self, node: onnx.NodeProto) -> None:
        super().visit_node(node)
        for i in range(len(node.input)):
            input = self.get_input(node, i)
            if input is not None and input.is_copy():
                node.input[i] = input.symbolic_value  # type: ignore[assignment]

        if is_onnx_op(node, "Identity"):
            input = self.get_input(node, 0)
            output = self.get_output(node, 0)
            if input is not None and output is not None:
                output.symbolic_value = input.name


# TODO: "Z = Identity(x)" where Z is a graph-output cannot be handled by this optimization,
# and requires some extension. (Eg., we could rename graph-output to be Z or we can try to
# rename x to be Z.)


def get_node_attr_value(node: onnx.NodeProto, attr_name: str, default: Any) -> Any:
    matching = [x for x in node.attribute if x.name == attr_name]
    if len(matching) > 1:
        raise ValueError(f"Node has multiple attributes with name {attr_name}")
    if len(matching) < 1:
        return default
    return onnx.helper.get_attribute_value(matching[0])


class SymbolicEvaluator(CopyPropagator):
    def __init__(self):
        super().__init__()

    def visit_node(self, node: onnx.NodeProto) -> None:
        super().visit_node(node)

        if is_onnx_op(node, "SequenceConstruct"):
            output = self.get_output(node, 0)
            if output is not None:
                output.symbolic_value = list(node.input)

        if is_onnx_op(node, "ConcatFromSequence"):
            input = self.get_input(node, 0)
            new_axis = get_node_attr_value(node, "new_axis", 0)
            if input is not None and isinstance(input.symbolic_value, list) and new_axis == 0:
                node.op_type = "Concat"
                node.input[:] = input.symbolic_value
                for i in range(len(node.attribute)):
                    if node.attribute[i].name == "new_axis":
                        del node.attribute[i]
                        break

        # TODO: handle SequenceEmpty, SequenceAt, etc.


def do_copy_propagation(model: onnx.ModelProto, *, remove_unused: bool = True) -> None:
    transformer = CopyPropagator()
    transformer.visit_model(model)
    if remove_unused:
        onnxscript.optimizer.remove_unused_nodes(model)


def do_sequence_simplification(model: onnx.ModelProto, *, remove_unused: bool = True) -> None:
    transformer = SymbolicEvaluator()
    transformer.visit_model(model)
    if remove_unused:
        onnxscript.optimizer.remove_unused_nodes(model)
