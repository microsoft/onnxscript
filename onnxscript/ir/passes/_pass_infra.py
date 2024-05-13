# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Passes infrastructure for the IR."""

from __future__ import annotations

import logging

__all__ = ["PassBase", "NodeTransformer"]

import abc

from onnxscript import ir

# This module implements NodeTransformer API described in
# https://pytorch.org/executorch/stable/compiler-custom-compiler-passes.html
# for the ONNX IR.

logger = logging.getLogger(__name__)


class PassBase(abc.ABC):
    """Base class for all passes.

    Class attributes:
        in_place: Whether the pass modifies the model in place.
    """

    in_place: bool = True

    def __call__(self, model: ir.Model) -> ir.Model:
        self.requires(model)
        result = self.call(model)
        self.ensures(model)
        return result

    @abc.abstractmethod
    def call(self, model: ir.Model) -> ir.Model:
        """The main entry point for the pass."""
        ...

    def requires(self, model: ir.Model) -> None:
        """Pre-conditions for the pass.

        This is optional to implement, will be called before call().
        """
        del model  # Unused

    def ensures(self, model: ir.Model) -> None:
        """Post-conditions for the pass.

        This is optional to implement, will be called after call().
        """
        del model  # Unused


class NodeTransformer(PassBase):
    """NodeTransformer for the ONNX IR.

    An NodeTransformer is a pass that traverses the IR and performs some
    operation on the nodes. The operation can be anything, such as
    checking invariants, transforming the IR, or generating code.

    By default, the NodeTransformer updates the model in place.

    Attributes:
        _model: ir.Model: The model being interpreted.
        scope (list[ir.Graph]): The current graph the NodeTransformer is running on.
    """

    def __init__(self, reversed: bool = False):
        self._model: ir.Model | None = None
        self.scope: list[ir.Graph] = []
        self.reversed = reversed

    def call(self, model: ir.Model) -> ir.Model:
        self._model = model
        self._call_graph(self._model.graph)
        return model

    def _call_graph(self, graph: ir.Graph):
        self.enter_graph(graph)
        self.scope.append(graph)
        iterable = reversed(graph) if self.reversed else graph
        for node in iterable:
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
