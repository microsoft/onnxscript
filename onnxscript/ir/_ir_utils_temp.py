"""This is a temporary utility to assist new IR while it's still under development."""

from __future__ import annotations

import numpy as np

from onnxscript import ir
from onnxscript.ir import serde


def propagate_const_value(ir_value: ir.Value) -> ir.Value:
    node = ir_value.def_node()
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
        if isinstance(constant_value, serde.TensorProtoTensor):
            return constant_value.numpy()
        return np.array(constant_value)
    return constant_value


# TODO: This is a temporary utility to assist new ir.Value naming
GEN_VAR_COUNTER: int = 0


def _make_new_name() -> str:
    global GEN_VAR_COUNTER  # pylint: disable=global-statement
    GEN_VAR_COUNTER += 1
    return f"_gen_{GEN_VAR_COUNTER}"


def post_node_output_naming(node: ir.Node) -> None:
    for output in node.outputs:
        assert output.name is None
        output.name = _make_new_name()
