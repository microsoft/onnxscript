# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Abstract backend for where newly created nodes and initializers are stored.

The rewrite engine instantiates a concrete ``NodeSink``, passes it into
``RewriterContext``, and after the rule returns, harvests the results via
the sink's properties.  This decoupling allows the storage backend to be
swapped (e.g., from a list-based tape to a live ``ir.Graph``) without
changing any rule code.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

import onnx_ir as ir

UsedOpsets = set[tuple[str, Optional[int]]]


class NodeSink(abc.ABC):
    """Abstract interface for where newly created nodes and initializers go.

    Concrete implementations define the storage strategy (list-based tape,
    graph-based, etc.).  The rewrite engine harvests results via the
    ``nodes``, ``initializers``, and ``used_opsets`` properties.
    """

    @abc.abstractmethod
    def add_node(self, node: ir.Node) -> None:
        """Record a newly created node."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_initializer(self, value: ir.Value) -> None:
        """Record a newly created initializer."""
        raise NotImplementedError

    @abc.abstractmethod
    def record_opset(self, domain: str, version: int | None) -> None:
        """Record that an opset domain/version was referenced."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nodes(self) -> Sequence[ir.Node]:
        """All nodes created during this replacement."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def initializers(self) -> Sequence[ir.Value]:
        """All initializers created during this replacement."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def used_opsets(self) -> UsedOpsets:
        """Opset domains/versions referenced by created nodes."""
        raise NotImplementedError


class TapeSink(NodeSink):
    """Append-only sink backed by plain lists.

    This wraps the same storage semantics as ``onnx_ir.tape.Tape`` but as a
    standalone object that can be passed into ``RewriterContext`` via composition.
    """

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []
        self._used_opsets: UsedOpsets = set()

    def add_node(self, node: ir.Node) -> None:
        self._nodes.append(node)

    def add_initializer(self, value: ir.Value) -> None:
        self._initializers.append(value)

    def record_opset(self, domain: str, version: int | None) -> None:
        self._used_opsets.add((domain, version))

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    @property
    def initializers(self) -> Sequence[ir.Value]:
        return tuple(self._initializers)

    @property
    def used_opsets(self) -> UsedOpsets:
        return self._used_opsets
