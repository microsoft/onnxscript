"""This is a temporary utility to assist new IR while it's still under development."""

from __future__ import annotations

import numpy as np

from onnxscript import ir

GRAPH_OUTPUT_META_KEY = "pkg.onnxscript.rewriter.generic_pattern.graph_output"


def propagate_const_value(ir_value: ir.Value) -> ir.Value:
    node = ir_value.producer()
    if ir_value.const_value is None and node is not None and node.op_type == "Constant":
        attr_names = [
            "value_float",
            "value_int",
            "value_string",
            "value",
            "value_floats",
            "value_ints",
            "value_strings",
        ]
        for attr_name in attr_names:
            attr_value = node.attributes.get(attr_name)
            if attr_value is not None:
                # TODO: RefAttr should be also supported?
                if isinstance(attr_value, ir.Attr):
                    ir_value.const_value = attr_value.value  # type: ignore[union-attr]
                    break
    return ir_value


def get_numpy_from_ir_value(value: ir.Value) -> np.ndarray | None:
    constant_value = value.const_value
    if constant_value is not None:
        if isinstance(constant_value, ir.serde.TensorProtoTensor):
            return constant_value.numpy()
        return np.array(constant_value)
    return constant_value
