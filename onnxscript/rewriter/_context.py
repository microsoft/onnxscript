# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Separated interfaces for the rewriter's node-building context.

This module defines two interfaces:

- ``RewriterContext``: The restricted API exposed to rewrite-rule authors.
  It allows creating new nodes (via ``op.OpName(...)`` or ``op.op(...)``) and
  new initializers (via ``op.initializer(...)``), but does **not** expose the
  accumulated nodes, initializers, or opset metadata.

- ``TapeContext``: The full implementation used internally by the rewrite engine.
  It extends ``RewriterContext`` with properties for harvesting the nodes,
  initializers, and opset information produced during a replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

from onnx_ir import tape

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Rule-facing API (restricted)
# ---------------------------------------------------------------------------


@runtime_checkable
class RewriterContext(Protocol):
    """The interface available to rewrite-rule ``rewrite()`` functions.

    Rewrite rules receive an object conforming to this protocol as the ``op``
    parameter.  It supports three operations:

    1. **Dynamic op dispatch** — ``op.Relu(x)``, ``op.MatMul(a, b, _domain=...)``, etc.
    2. **Explicit op creation** — ``op.op("Conv", inputs, attrs, domain=...)``.
    3. **Initializer creation** — ``op.initializer(tensor, name=...)``.

    Rules must **not** access implementation details such as the list of nodes
    or initializers created so far; those are reserved for the rewrite engine.
    """

    def __getattr__(self, op_type: str) -> Any:
        """Dynamic op dispatch.

        Returns a callable that creates a node of the given ``op_type`` and
        records it on the internal tape.

        Supported keyword arguments:
            _domain (str): Op domain (default ``""``).
            _version (int | None): Opset version.
            _outputs (int | list[str]): Number of outputs or explicit output names.
            _name (str | None): Optional node name (must be unique).
        """
        ...

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, Any] | None = None,
        *,
        domain: str = "",
        version: int | None = None,
        name: str | None = None,
    ) -> ir.Value:
        """Create a single-output node with an explicit op type.

        This is useful when the op type is determined dynamically or when
        forwarding attributes from a matched node.
        """
        ...

    def initializer(
        self,
        tensor: ir.TensorProtocol,
        name: str | None = None,
    ) -> ir.Value:
        """Create a new constant initializer and return its ``ir.Value``."""
        ...


# ---------------------------------------------------------------------------
# Engine-facing implementation (full access)
# ---------------------------------------------------------------------------


class TapeContext(tape.Tape):
    """Full builder used internally by the rewrite engine.

    Rules receive this object typed as :class:`RewriterContext`, so they see
    only the restricted protocol.  The engine retains the concrete
    ``TapeContext`` reference and can access ``.nodes``, ``.initializers``,
    and ``.used_opsets`` to harvest the replacement subgraph.

    This class inherits from :class:`onnx_ir.tape.Tape` (via
    :class:`onnxscript.ir._tape.Builder`) and adds the convenient
    ``__getattr__``-based dynamic op dispatch (``op.Relu(x)``).
    """

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(
        self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]
    ) -> ir.Value | Sequence[ir.Value]:
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
            value = super().op(
                op_type,
                inputs=inputs,
                attributes=kwargs,
                domain=domain,
                version=version,
                name=name,
            )
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = super().op_multi_out(
            op_type,
            inputs=inputs,
            attributes=kwargs,
            domain=domain,
            version=version,
            name=name,
            num_outputs=num_outputs,
        )
        if isinstance(outputs, Sequence):
            for value, output_name in zip(values, outputs):
                value.name = output_name
        return values

    # The following properties are inherited from Tape and are intentionally
    # NOT part of the RewriterContext protocol.  They are used only by the
    # rewrite engine to harvest the replacement subgraph:
    #
    #   .nodes          -> Sequence[ir.Node]
    #   .initializers   -> Sequence[ir.Value]
    #   .used_opsets    -> UsedOpsets
