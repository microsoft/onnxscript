# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Convenience methods for constructing the IR."""

from __future__ import annotations

from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from onnxscript import ir
from onnxscript.ir import _convenience

# A type representing the domains/versions used in creating nodes in IR.
UsedOpsets = set[Tuple[str, Optional[int]]]


class Tape:
    """Tape class.

    A tape is a recorder that collects nodes and initializers that are created so
    that they can be used for creating a graph.

    Example::

        from onnxscript import ir

        tape = ir.tape.Tape()
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

    Attributes:
        graph_like: The graph to append the new nodes and initializers to. When
            it is None, the nodes and initializers are creating without owned by a graph.
            Initializers will not be added to functions because it is not supported by ONNX.
    """

    def __init__(self, graph_like: ir.Graph | ir.Function | None = None) -> None:
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []
        self._used_opsets: UsedOpsets = set()
        self.graph_like = graph_like

    def __repr__(self) -> str:
        return f"Tape(nodes={self._nodes}, initializers={self._initializers})"

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    @property
    def initializers(self) -> Sequence[ir.Value]:
        return tuple(self._initializers)

    @property
    def used_opsets(self) -> UsedOpsets:
        return self._used_opsets

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
            graph=graph or self.graph_like,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        self._nodes.append(node)
        self._used_opsets.add((domain, version))

        return node.outputs[0]

    def op_multi_output(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, _convenience.SupportedAttrTypes] | None = None,
        *,
        num_outputs: int,
        domain: str = "",
        overload: str = "",
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
            graph=graph or self.graph_like,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        self._nodes.append(node)
        self._used_opsets.add((domain, version))

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
        if isinstance(self.graph_like, ir.Graph):
            self.graph_like.register_initializer(value)
        return value


class Builder(Tape):
    """An extension of the tape that provides a more convenient API for constructing the IR."""

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

        if num_outputs == 1:
            value = super().op(
                op_type, inputs=inputs, attributes=kwargs, domain=domain, version=version
            )
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = super().op_multi_output(
            op_type,
            inputs=inputs,
            attributes=kwargs,
            domain=domain,
            version=version,
            num_outputs=num_outputs,
        )
        if isinstance(outputs, Sequence):
            for value, name in zip(values, outputs):
                value.name = name
        return values
