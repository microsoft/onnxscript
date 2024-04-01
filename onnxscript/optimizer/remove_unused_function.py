from __future__ import annotations

import logging

import onnx
from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
)

logger = logging.getLogger(__name__)


class UnusedFunctionRemover:
    def compute_used_in_node(self, n: onnx.NodeProto) -> set[tuple[str, str]]:
        used = {(n.domain, n.op_type)}
        for attr in n.attribute:
            if attr.HasField("g"):
                used |= self.process_graph(attr.g)
            elif len(attr.graphs) > 0:
                for graph in attr.graphs:
                    used |= self.process_graph(graph)
        if (n.domain, n.op_type) in self._functions:
            function = self._functions[(n.domain, n.op_type)]
            used |= self.process_function(function)
        return used

    def process_nodes(
        self, nodes: RepeatedCompositeFieldContainer[onnx.NodeProto]
    ) -> set[tuple[str, str]]:
        used = set()
        for node in nodes:
            used |= self.compute_used_in_node(node)
        return used

    def process_graph(self, graph: onnx.GraphProto) -> set[tuple[str, str]]:
        return self.process_nodes(graph.node)

    def process_function(self, function: onnx.FunctionProto) -> set[tuple[str, str]]:
        return self.process_nodes(function.node)

    def process_model(self, model: onnx.ModelProto) -> None:
        self._functions = {(f.domain, f.name): f for f in model.functions}
        used = self.process_graph(model.graph)
        count = 0
        logger.debug("Used function protos: %s", used)
        for i in range(len(model.functions) - 1, -1, -1):
            if (model.functions[i].domain, model.functions[i].name) not in used:
                del model.functions[i]
                count += 1
        logger.info("Removed %s unused function protos", count)
        logger.debug("Function protos left: %s", [f.name for f in model.functions])


def remove_unused_functions(model: onnx.ModelProto) -> None:
    """Removes unused function protos from the model."""
    UnusedFunctionRemover().process_model(model)
