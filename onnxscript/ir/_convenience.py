# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convenience methods for constructing and manipulating the IR.

This is an internal only module. We should choose to expose some of the methods
after they are proven to be useful.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import onnx

from onnxscript.ir import _core, _enums, _protocols, serde

SupportedAttrTypes = Union[
    str,
    int,
    float,
    Sequence[int],
    Sequence[float],
    Sequence[str],
    _protocols.TensorProtocol,
    onnx.TensorProto,
    _core.Attr,
    None,
]


def convert_attribute(
    name: str,
    attr: SupportedAttrTypes,
    attr_type: _enums.AttributeType | None = None,
) -> _core.Attr:
    """Convert a Python object to a _core.Attr object.

    This method is useful when constructing nodes with attributes. It infers the
    attribute type based on the type of the Python value.

    Args:
        name: The name of the attribute.
        attr: The value of the attribute.
        attr_type: The type of the attribute. This is required when attr is None.

    Returns:
        A ``Attr`` object.
    """
    if attr is None:
        if attr_type is None:
            raise ValueError("attr_type must be provided when attr is None")
        return _core.Attr(name, attr_type, None)
    if isinstance(attr, int):
        return _core.AttrInt64(name, attr)
    if isinstance(attr, float):
        return _core.AttrFloat32(name, attr)
    if isinstance(attr, str):
        return _core.AttrString(name, attr)
    if isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
        return _core.AttrInt64s(name, attr)  # type: ignore
    if isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
        return _core.AttrFloat32s(name, attr)  # type: ignore
    if isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
        return _core.AttrStrings(name, attr)  # type: ignore
    if isinstance(attr, (_core.Tensor, _protocols.TensorProtocol)):
        return _core.AttrTensor(name, attr)
    if isinstance(attr, onnx.TensorProto):
        return _core.AttrTensor(name, serde.TensorProtoTensor(attr))
    if isinstance(attr, _core.Attr):
        return attr
    raise TypeError(f"Unsupported attribute type: '{type(attr)}'")


def convert_attributes(attrs: Mapping[str, SupportedAttrTypes]) -> list[_core.Attr]:
    """Convert a dictionary of attributes to a list of _core.Attr objects.

    It infers the attribute type based on the type of the value. The supported
    types are: int, float, str, Sequence[int], Sequence[float], Sequence[str],
    :class:`_core.Tensor`, and :class:`_core.Attr`.

    Args:
        attrs: A dictionary of {<attribute name>: <python objects> to convert.

    Returns:
        A list of _core.Attr objects.
    """
    attributes: list[_core.Attr] = []
    for name, attr in attrs.items():
        attributes.append(convert_attribute(name, attr))
    return attributes


def replace_all_uses_with(
    values: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol],
    replacements: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol],
) -> None:
    """Replace all consumers of the given values with the replacements.

    This is useful when nodes in the graph are replaced with new nodes, where
    the old users need to be updated to use the outputs of the new nodes.

    For example, suppose we have the following graph::

        A -> {B, C}

    We want to replace the node A with a new node D::

        >>> from onnxscript import ir
        >>> input = ir.Input("input")
        >>> node_a = ir.Node("", "A", [input])
        >>> node_b = ir.Node("", "B", node_a.outputs)
        >>> node_c = ir.Node("", "C", node_a.outputs)
        >>> node_d = ir.Node("", "D", [input])
        >>> replace_all_uses_with(node_a.outputs, node_d.outputs)
        >>> len(node_b.inputs)
        1
        >>> node_b.inputs[0].producer().op_type
        'D'
        >>> len(node_c.inputs)
        1
        >>> node_c.inputs[0].producer().op_type
        'D'
        >>> len(node_a.outputs[0].consumers())
        0

    When values and replacements are sequences, they are zipped into pairs. All
    users of the first value is replaced with the first replacement, and so on.

    .. note::
        You still need to update the graph outputs if any of the values being
        replaced are part of the graph outputs. Be sure to remove the old nodes
        from the graph using ``graph.remove()`` if they are no longer needed.

    Args:
        values: The value or values to be replaced.
        replacements: The new value or values to use as inputs.
    """
    if not isinstance(values, Sequence):
        values = (values,)
    if not isinstance(replacements, Sequence):
        replacements = (replacements,)
    if len(values) != len(replacements):
        raise ValueError("The number of values and replacements must match.")
    for value, replacement in zip(values, replacements):
        for user_node, index in tuple(value.consumers()):
            user_node.replace_input_with(index, replacement)
