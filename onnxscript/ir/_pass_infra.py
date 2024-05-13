"""Passes infrastructure for the IR."""

from __future__ import annotations

__all__ = ["PassBase", "Interpreter"]

import abc

from onnxscript import ir

# This module implements interpreter API described in
# https://pytorch.org/executorch/stable/compiler-custom-compiler-passes.html
# for the ONNX IR.

class PassBase(abc.ABC):
    """Base class for all passes."""

    def __call__(self, model: ir.Model):
        self.requires(model)
        result = self.call(model)
        self.ensures(model)
        return result

    @abc.abstractmethod
    def call(self, model: ir.Model):
        """The main entry point for the pass."""
        ...

    def requires(self, model: ir.Model) -> None:
        """Preconditions for the pass.

        This is optional to implement, will be called before call().
        """
        del model  # Unused

    def ensures(self, model: ir.Model) -> None:
        """Postconditions for the pass.

        This is optional to implement, will be called after call().
        """
        del model  # Unused


class Interpreter(PassBase):
    """Interpreter for the ONNX IR.

    An interpreter is a pass that traverses the IR and performs some
    operation on the nodes. The operation can be anything, such as
    checking invariants, transforming the IR, or generating code.

    Attributes:
        _model: ir.Model: The model being interpreted.
        scope (list[ir.Graph]): The current graph the interpreter is running on.
    """
    def __init__(self):
        self._model: ir.Model | None = None
        self.scope: list[ir.Graph] = []

    def call(self, model: ir.Model):
        self._model = model
        self._call_graph(self._model.graph)

    def _call_graph(self, graph: ir.Graph):
        self.enter_graph(graph)
        self.scope.append(graph)
        for node in graph:
            self._call_node(node)
        self.exit_graph(graph)
        self.scope.pop()

    def _call_node(self, node: ir.Node):
        self.call_node(node)
        for attr in node.attributes.values():
            if not isinstance(attr, ir.Attr):
                continue
            if attr.type == ir.AttributeType.GRAPH:
                self._call_graph(attr.value)
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self._call_graph(graph)

    def enter_graph(self, graph: ir.Graph):
        """Called when entering a graph. Optional to implement."""
        del graph  # Unused

    def exit_graph(self, graph: ir.Graph):
        """Called when exiting a graph. Optional to implement."""
        del graph  # Unused

    @abc.abstractmethod
    def call_node(self, node: ir.Node):
        """Called when visiting a node."""
        ...
