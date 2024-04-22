"""Convenience methods for constructing the IR."""

# NOTE: This is a temporary solution for constructing the IR. It should be replaced
# with a more permanent solution in the future.

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from onnxscript import ir
from onnxscript.ir import _convenience


class Tape(Iterable[ir.Node]):
    """A tape for recording nodes that are created."""

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []

    def __iter__(self) -> Sequence[ir.Node]:
        return self._nodes

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        domain: str = "",
    ) -> ir.Value:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = _convenience.convert_attributes(attributes)
        node = ir.Node(domain, op_type, inputs, attributes=attrs, num_outputs=1)
        self._nodes.append(node)

        return node.outputs[0]

    def op_multi_output(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        num_outputs: int,
        domain: str = "",
    ) -> Sequence[ir.Value]:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = _convenience.convert_attributes(attributes)
        node = ir.Node(domain, op_type, inputs, attributes=attrs, num_outputs=num_outputs)
        self._nodes.append(node)

        return node.outputs
