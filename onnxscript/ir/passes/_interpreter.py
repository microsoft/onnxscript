"""Passes infrastructure for the IR."""

import abc

from onnxscript import ir

# This module implements interpreter API described in
# https://pytorch.org/executorch/stable/compiler-custom-compiler-passes.html
# for the ONNX IR.


class Interpreter(abc.ABC):
    def __init__(self):
        self._model = None
        self.scope = []

    def __call__(self, model: ir.Model):
        self._model = model
        self.call_model(self._model)
        self._call_graph(self._model.graph)

    def _call_graph(self, graph: ir.Graph):
        self.enter_graph(graph)
        self.scope.append(graph)
        for node in graph:
            self._call_node(node)
        self.exit_graph(graph)
        self.scope.pop()

    def _call_node(self, node: ir.Node):
        self.run_node(node)
        for attr in node.attributes.values():
            if attr.type == ir.AttributeType.GRAPH and isinstance(attr, ir.Attr):
                self._call_graph(attr.value)
            elif attr.type == ir.AttributeType.GRAPHS and isinstance(attr, ir.Attr):
                for graph in attr.value:
                    self._call_graph(graph)

    @abc.abstractmethod
    def call_model(self, model: ir.Model):
        return

    @abc.abstractmethod
    def enter_graph(self, graph: ir.Graph):
        return

    @abc.abstractmethod
    def exit_graph(self, graph: ir.Graph):
        return

    @abc.abstractmethod
    def run_node(self, node: ir.Node):
        return
