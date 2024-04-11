"""This is a temporary utility to assist new IR while it's still under development."""

from __future__ import annotations

from onnxscript import ir
from onnxscript.ir import serde
import numpy as np


def propogate_const_value(value: ir.Value) -> ir.Value:
    node = value.def_node()
    if value.const_value is None and node is not None and node.op_type == "Constant":
        if (
            ((attr_value := node.attributes.get("value_float")) is not None)
            or ((attr_value := node.attributes.get("value_int")) is not None)
            or ((attr_value := node.attributes.get("value_string")) is not None)
            or ((attr_value := node.attributes.get("value")) is not None)
            or ((attr_value := node.attributes.get("value_floats")) is not None)
            or ((attr_value := node.attributes.get("value_ints")) is not None)
            or ((attr_value := node.attributes.get("value_strings")) is not None)
        ):
            value.const_value = attr_value.value
    return value

def get_numpy_from_ir_value(value: ir.Value) -> np.ndarray | None:
    constant_value = value.const_value
    if constant_value is not None:
        if isinstance(constant_value, serde.TensorProtoTensor):
            constant_value = constant_value.numpy()
        else:
            constant_value = np.array(constant_value)
    return constant_value

GEN_VAR_COUNTER: int = 0

def _make_new_name() -> str:
    global GEN_VAR_COUNTER
    GEN_VAR_COUNTER += 1
    return f"_gen_{GEN_VAR_COUNTER}"

def post_node_output_naming(node: ir.Node) -> None:
    for output in node.outputs:
        assert output.name is None
        output.name = _make_new_name()
