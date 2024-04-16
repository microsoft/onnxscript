"""Convenience methods for constructing the IR."""

from __future__ import annotations

import collections.abc
from typing import Any, Mapping, Sequence

from onnxscript import ir


def _convert_attributes(attrs: Mapping[str, Any]) -> list[ir.Attr]:
    attributes = []
    for name, attr in attrs.items():
        if isinstance(attr, int):
            attributes.append(ir.AttrInt64(name, attr))
        elif isinstance(attr, float):
            attributes.append(ir.AttrFloat32(name, attr))
        elif isinstance(attr, str):
            attributes.append(ir.AttrString(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
            attributes.append(ir.AttrInt64s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
            attributes.append(ir.AttrFloat32s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
            attributes.append(ir.AttrStrings(name, attr))
        elif isinstance(attr, ir.Attr):
            attributes.append(attr)
        else:
            raise TypeError(f"Unsupported attribute type: '{type(attr)}'")
    return attributes


class Tape(collections.abc.Iterable[ir.Node]):
    """A tape for recording nodes that are created."""

    def __init__(self) -> None:
        self._nodes = []

    def __iter__(self) -> Sequence[ir.Node]:
        return self._nodes

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, Any] | None = None,
        domain: str = "",
    ) -> ir.Value:
        if attributes is None:
            attrs = ()
        else:
            attrs = _convert_attributes(attributes)
        node = ir.Node(domain, op_type, inputs, attributes=attrs, num_outputs=1)
        self._nodes.append(node)

        return node.outputs[0]

    def op_multi_output(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, Any] | None = None,
        *,
        num_outputs: int,
        domain: str = "",
    ) -> Sequence[ir.Value]:
        if attributes is None:
            attrs = ()
        else:
            attrs = _convert_attributes(attributes)
        node = ir.Node(
            domain, op_type, inputs, attributes=attrs, num_outputs=num_outputs
        )
        self._nodes.append(node)

        return node.outputs
