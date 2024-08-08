# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing the IR."""

# NOTE: This is a temporary solution for constructing the IR. It should be replaced
# with a more permanent solution in the future.

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from onnxscript import ir
from onnxscript.ir import _convenience


class Tape(Iterable[ir.Node]):
    """A tape for recording nodes that are created."""

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []

    def __iter__(self) -> Iterator[ir.Node]:
        return iter(self._nodes)

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


# A type representing the domains/versions used in creating nodes in IR.
UsedOpsets = List[Tuple[str, Optional[int]]]


class Builder(Tape):
    """An extension of the tape that provides a more convenient API for constructing the IR."""

    def __init__(self):
        super().__init__()
        self._used_opsets: UsedOpsets = []

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)
        if isinstance(outputs, Sequence):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, int)
            num_outputs = outputs

        self._used_opsets.append((domain, version))
        if num_outputs == 1:
            value = super().op(op_type, inputs=inputs, attributes=kwargs, domain=domain)
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = super().op_multi_output(
            op_type, inputs=inputs, attributes=kwargs, domain=domain, num_outputs=num_outputs
        )
        if isinstance(outputs, Sequence):
            for value, name in zip(values, outputs):
                value.name = name
        return values

    @property
    def used_opsets(self) -> UsedOpsets:
        return self._used_opsets
