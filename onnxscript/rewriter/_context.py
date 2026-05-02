# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Separated interfaces for the rewriter's node-building context.

This module defines:

- ``RewriterContext``: The restricted API exposed to rewrite-rule authors.
  It allows creating new nodes (via ``op.OpName(...)`` or ``op.op(...)``) and
  new initializers (via ``op.initializer(...)``), but blocks access to internal
  state such as the accumulated nodes, initializers, or opset metadata.

- ``NodeSink``: The abstract backend that stores nodes and initializers.
  The engine creates a sink, passes it to ``RewriterContext``, and harvests
  results from the sink after the rule returns.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import onnx_ir as ir
from onnx_ir import _convenience

from onnxscript.rewriter._node_sink import NodeSink

# Attribute names that rules must NOT access.  We block both the public
# harvesting properties and common private names to prevent accidental
# coupling to the implementation.
_FORBIDDEN_ATTRS = frozenset(
    {
        "nodes",
        "initializers",
        "used_opsets",
        "_nodes",
        "_initializers",
        "_used_opsets",
        "graph_like",
        "_sink",
    }
)


class RewriterContext:
    """The interface available to rewrite-rule ``rewrite()`` functions.

    Rewrite rules receive an instance of this class as the ``op`` parameter.
    It supports three operations:

    1. **Dynamic op dispatch** — ``op.Relu(x)``, ``op.MatMul(a, b, _domain=...)``, etc.
    2. **Explicit op creation** — ``op.op("Conv", inputs, attrs, domain=...)``.
    3. **Initializer creation** — ``op.initializer(tensor, name=...)``.

    Accessing engine-internal state (``nodes``, ``initializers``, ``used_opsets``,
    etc.) raises ``AttributeError`` with a descriptive message.

    Args:
        sink: The :class:`NodeSink` backend where created nodes and initializers
            are stored.  The engine retains a reference to this sink and harvests
            results after the rule returns.
    """

    def __init__(self, sink: NodeSink) -> None:
        # Store the sink directly on the instance dict, bypassing __getattribute__.
        object.__setattr__(self, "_RewriterContext__sink", sink)

    def __getattribute__(self, name: str) -> Any:
        if name in _FORBIDDEN_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}.{name}' is not available to rewrite rules. "
                f"Use op.OpName(...), op.op(...), or op.initializer(...) only."
            )
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        """Dynamic op dispatch: ``op.Relu(x)``, ``op.MatMul(a, b)``, etc.

        Returns a callable that creates a node of the given ``name`` (as op_type)
        and records it on the internal sink.

        Accessing engine-internal names (``nodes``, ``initializers``, etc.) raises
        ``AttributeError``.

        Supported keyword arguments on the returned callable:
            _domain (str): Op domain (default ``""``).
            _version (int | None): Opset version.
            _outputs (int | list[str]): Number of outputs or explicit output names.
            _name (str | None): Optional node name (must be unique).
        """
        if name in _FORBIDDEN_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}.{name}' is not available to rewrite rules. "
                f"Use op.OpName(...), op.op(...), or op.initializer(...) only."
            )
        return lambda *args, **kwargs: self._make_node(name, args, kwargs)

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

        if num_outputs == 1:
            value = self._create_single_output_node(
                op_type, inputs, kwargs, domain=domain, version=version, name=name
            )
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value

        values = self._create_multi_output_node(
            op_type,
            inputs,
            kwargs,
            domain=domain,
            version=version,
            name=name,
            num_outputs=num_outputs,
        )
        if isinstance(outputs, Sequence):
            for value, output_name in zip(values, outputs):
                value.name = output_name
        return values

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
        return self._create_single_output_node(
            op_type, inputs, attributes, domain=domain, version=version, name=name
        )

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
        sink: NodeSink = object.__getattribute__(self, "_RewriterContext__sink")
        sink.add_initializer(value)
        return value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_single_output_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        domain: str = "",
        version: int | None = None,
        name: str | None = None,
    ) -> ir.Value:
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
        sink: NodeSink = object.__getattribute__(self, "_RewriterContext__sink")
        sink.add_node(node)
        sink.record_opset(domain, version)
        return node.outputs[0]

    def _create_multi_output_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        domain: str = "",
        version: int | None = None,
        name: str | None = None,
        num_outputs: int,
    ) -> Sequence[ir.Value]:
        attrs: Sequence[ir.Attr] = (
            _convenience.convert_attributes(attributes) if attributes else ()
        )
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            num_outputs=num_outputs,
            version=version,
            name=name,
        )
        sink: NodeSink = object.__getattribute__(self, "_RewriterContext__sink")
        sink.add_node(node)
        sink.record_opset(domain, version)
        return node.outputs
