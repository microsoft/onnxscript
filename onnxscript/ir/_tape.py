# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing the IR."""

from __future__ import annotations

from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from onnxscript import ir
from onnxscript.ir import _convenience, _core


class Tape(Iterable[_core.Node]):
    """Tape class.

    A tape is a recorder that collects nodes and initializers that are created so
    that they can be used for creating a graph.

    Example::
        from onnxscript import ir

        tape = Tape()
        a = tape.initializer(ir.tensor([1, 2, 3], name="a"))
        b: ir.Value = ...
        c: ir.Value = ...
        x = tape.op("Add", [a, b], attributes={"alpha": 1.0})
        y = tape.op("Mul", [x, c], attributes={"beta": 2.0})
        model = ir.Model(
            graph := ir.Graph(
                inputs=[b, c],
                outputs=[y],
                nodes=tape.nodes,
                initializers=tape.initializers
                opset_imports={"": 20},
            ),
            ir_version=10,
        )
    """

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []

    def __iter__(self) -> Iterator[ir.Node]:
        return iter(self._nodes)

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    @property
    def initializers(self) -> Sequence[ir.Value]:
        return tuple(self._initializers)

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        domain: str = "",
        overload: str = "",
        version: int | None = None,
        graph: ir.Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> ir.Value:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = _convenience.convert_attributes(attributes)
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            num_outputs=1,
            overload=overload,
            version=version,
            graph=graph,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        self._nodes.append(node)

        return node.outputs[0]

    def op_multi_output(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        domain: str = "",
        overload: str = "",
        num_outputs: int | None = None,
        version: int | None = None,
        graph: ir.Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> Sequence[ir.Value]:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = _convenience.convert_attributes(attributes)
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            num_outputs=num_outputs,
            overload=overload,
            version=version,
            graph=graph,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        self._nodes.append(node)

        return node.outputs

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        name = name or tensor.name
        if name is None:
            raise ValueError("Name must be provided for initializer.")
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._initializers.append(value)
        return value


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
