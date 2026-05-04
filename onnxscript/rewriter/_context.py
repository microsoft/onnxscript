# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Rewriter context for building replacement subgraphs in rewrite rules.

This module defines:

- ``OpBuilderBase``: Abstract base class exposing the rule-facing API for
  creating new nodes (via ``op.OpName(...)`` or ``op.op(...)``) and new
  initializers (via ``op.initializer(...)``).  Subclasses implement the
  storage strategy by overriding ``_add_node``, ``_add_initializer``, and
  ``_record_opset``.

- ``TapeRewriterContext``: Concrete subclass backed by simple lists.  The
  rewrite engine creates an instance, passes it to the rule, and harvests
  the accumulated nodes / initializers / opsets after the rule returns.

- ``RewriterContext``: Alias for ``OpBuilderBase`` (used by rewrite rules).
- ``OptimizerContext``: Alias for ``OpBuilderBase`` (used by the optimizer).
"""

from __future__ import annotations

import abc
from typing import Any, Mapping, Optional, Sequence

import onnx_ir as ir
from onnx_ir import _convenience

UsedOpsets = set[tuple[str, Optional[int]]]


class OpBuilderBase(abc.ABC):
    """The interface available to rewrite-rule ``rewrite()`` functions.

    Rewrite rules receive an instance of a concrete subclass as the ``op``
    parameter.  It supports three operations:

    1. **Dynamic op dispatch** — ``op.Relu(x)``, ``op.MatMul(a, b, _domain=...)``, etc.
    2. **Explicit op creation** — ``op.op("Conv", inputs, attrs, domain=...)``.
    3. **Initializer creation** — ``op.initializer(tensor, name=...)``.

    Subclasses must implement the three protected methods that define where
    created nodes and initializers are stored:

    - :meth:`_add_node`
    - :meth:`_add_initializer`
    - :meth:`_record_opset`
    """

    # ------------------------------------------------------------------
    # Abstract storage interface (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _add_node(self, node: ir.Node) -> None:
        """Record a newly created node."""
        raise NotImplementedError

    @abc.abstractmethod
    def _add_initializer(self, value: ir.Value) -> None:
        """Record a newly created initializer."""
        raise NotImplementedError

    @abc.abstractmethod
    def _record_opset(self, domain: str, version: int | None) -> None:
        """Record that an opset domain/version was referenced."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rule-facing API (concrete)
    # ------------------------------------------------------------------

    def __getattr__(self, op_type: str) -> Any:
        """Dynamic op dispatch: ``op.Relu(x)``, ``op.MatMul(a, b)``, etc.

        Returns a callable that creates a node of the given ``op_type``
        and records it via the subclass storage implementation.

        Supported keyword arguments on the returned callable:
            _domain (str): Op domain (default ``""``).
            _version (int | None): Opset version.
            _outputs (int | list[str]): Number of outputs or explicit output names.
            _name (str | None): Optional node name (must be unique).
        """
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(
        self, op_type: str, inputs: Sequence[ir.Value | None], kwargs: dict[str, Any]
    ) -> ir.Value | Sequence[ir.Value]:
        """Create one or more output values by building an ``ir.Node``."""
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)
        name = kwargs.pop("_name", None)

        if isinstance(outputs, Sequence):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, int)
            num_outputs = outputs

        attrs: Sequence[ir.Attr] = _convenience.convert_attributes(kwargs) if kwargs else ()
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            num_outputs=num_outputs,
            version=version,
            name=name,
        )
        self._add_node(node)
        self._record_opset(domain, version)

        if num_outputs == 1:
            if isinstance(outputs, Sequence):
                node.outputs[0].name = outputs[0]
            return node.outputs[0]

        if isinstance(outputs, Sequence):
            for value, output_name in zip(node.outputs, outputs):
                value.name = output_name
        return node.outputs

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        domain: str = "",
        version: int | None = None,
        name: str | None = None,
    ) -> ir.Value:
        """Create a single-output node with an explicit op type.

        This is useful when the op type is determined dynamically or when
        forwarding attributes from a matched node.
        """
        attrs: Sequence[ir.Attr] = (
            _convenience.convert_attributes(attributes) if attributes else ()
        )
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            num_outputs=1,
            version=version,
            name=name,
        )
        self._add_node(node)
        self._record_opset(domain, version)
        return node.outputs[0]

    def initializer(
        self,
        tensor: ir.TensorProtocol,
        name: str | None = None,
    ) -> ir.Value:
        """Create a new constant initializer and return its ``ir.Value``."""
        name = name or tensor.name
        if name is None:
            raise ValueError("Name must be provided for initializer.")
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._add_initializer(value)
        return value


class TapeRewriterContext(OpBuilderBase):
    """Concrete rewriter context backed by simple lists.

    The rewrite engine creates an instance, passes it to the rule's
    ``rewrite()`` function, and after the rule returns, harvests the
    accumulated results via the ``nodes``, ``initializers``, and
    ``used_opsets`` properties.
    """

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []
        self._used_opsets: UsedOpsets = set()

    def _add_node(self, node: ir.Node) -> None:
        self._nodes.append(node)

    def _add_initializer(self, value: ir.Value) -> None:
        self._initializers.append(value)

    def _record_opset(self, domain: str, version: int | None) -> None:
        self._used_opsets.add((domain, version))

    # --- Engine-only harvesting properties ---

    @property
    def nodes(self) -> Sequence[ir.Node]:
        """All nodes created during this replacement."""
        return tuple(self._nodes)

    @property
    def initializers(self) -> Sequence[ir.Value]:
        """All initializers created during this replacement."""
        return tuple(self._initializers)

    @property
    def used_opsets(self) -> UsedOpsets:
        """Opset domains/versions referenced by created nodes."""
        return self._used_opsets


# Public aliases for domain-specific usage
RewriterContext = OpBuilderBase
"""Alias for :class:`OpBuilderBase`, used in rewrite rule signatures."""

OptimizerContext = OpBuilderBase
"""Alias for :class:`OpBuilderBase`, used in optimizer partial-evaluator signatures."""
