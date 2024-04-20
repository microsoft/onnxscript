# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convenience methods for constructing and manipulating the IR.

This is an internal only module. We should choose to expose some of the methods
after they are proven to be useful.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from onnxscript.ir import _core, _protocols


def convert_attributes(attrs: Mapping[str, Any]) -> list[_core.Attr]:
    attributes: list[_core.Attr] = []
    for name, attr in attrs.items():
        if isinstance(attr, int):
            attributes.append(_core.AttrInt64(name, attr))
        elif isinstance(attr, float):
            attributes.append(_core.AttrFloat32(name, attr))
        elif isinstance(attr, str):
            attributes.append(_core.AttrString(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, int) for x in attr):
            attributes.append(_core.AttrInt64s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, float) for x in attr):
            attributes.append(_core.AttrFloat32s(name, attr))
        elif isinstance(attr, Sequence) and all(isinstance(x, str) for x in attr):
            attributes.append(_core.AttrStrings(name, attr))
        elif isinstance(attr, _core.Attr):
            attributes.append(attr)
        else:
            raise TypeError(f"Unsupported attribute type: '{type(attr)}'")
    return attributes


def replace_all_uses_with(values: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol], replacements: _protocols.ValueProtocol | Sequence[_protocols.ValueProtocol]) -> None:
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
        >>> node_b.inputs[0].op_type
        'D'
        >>> len(node_c.inputs)
        1
        >>> node_c.inputs[0].op_type
        'D'
        >>> len(node_a.consumers())
        0

    When values and replacements are sequences, they are zipped into pairs. All
    users of the first value is replaced with the first replacement, and so on.

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
        for user_node, index in tuple(replacement.consumers()):
            user_node.replace_input_with(index, value)
