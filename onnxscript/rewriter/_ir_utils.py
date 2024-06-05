# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This is a temporary utility to assist new IR while it's still under development."""

from __future__ import annotations

import typing

import numpy as np

from onnxscript import ir

GRAPH_OUTPUT_META_KEY = "pkg.onnxscript.rewriter.generic_pattern.graph_output"


def propagate_const_value(ir_value: ir.Value) -> ir.Value:
    """Temporary method to propagate a constant value to the IR value."""
    node = ir_value.producer()
    if node is None:
        return ir_value
    if node.op_type != "Constant":
        return ir_value
    attr_name, attr_value = next(iter(node.attributes.items()))
    if attr_value is None or not isinstance(attr_value, ir.Attr):
        return ir_value

    const_value: ir.TensorProtocol
    if attr_name in {"value_float", "value_floats"}:
        const_value = ir.Tensor(
            np.array(attr_value.value, dtype=np.float32), name=ir_value.name
        )
    elif attr_name in {"value_int", "value_ints"}:
        const_value = ir.Tensor(np.array(attr_value.value, dtype=np.int64), name=ir_value.name)
    elif attr_name in {"value_string", "value_strings"}:
        const_value = ir.StringTensor(
            np.array(attr_value.value, dtype=np.bytes_), name=ir_value.name
        )
    elif attr_name == "value":
        const_value = typing.cast(ir.TensorProtocol, attr_value.value)
    else:
        return ir_value

    ir_value.const_value = const_value
    ir_value.shape = const_value.shape  # type: ignore
    ir_value.dtype = const_value.dtype
    return ir_value
