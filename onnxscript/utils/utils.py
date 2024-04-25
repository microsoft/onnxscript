from __future__ import annotations

from typing import Any

import onnx


def normalize_domain(d: str) -> str:
    return "" if d == "ai.onnx" else d


def is_onnx_domain(d: str) -> bool:
    return normalize_domain(d) == ""


def is_onnx_op(node: onnx.NodeProto, op_type: str) -> bool:
    return is_onnx_domain(node.domain) and node.op_type == op_type


def is_control_flow_op(node: onnx.NodeProto) -> bool:
    return any(attr.HasField("g") or len(attr.graphs) > 0 for attr in node.attribute)


def get_node_attr_value(node: onnx.NodeProto, attr_name: str, default: Any) -> Any:
    matching = [x for x in node.attribute if x.name == attr_name]
    if len(matching) > 1:
        raise ValueError(f"Node has multiple attributes with name {attr_name}")
    if len(matching) < 1:
        return default
    return onnx.helper.get_attribute_value(matching[0])


def get_initializer_type(initializer: onnx.TensorProto) -> onnx.TypeProto:
    type = onnx.TypeProto()
    type.tensor_type.elem_type = initializer.data_type
    dims = type.tensor_type.shape.dim
    for dim in initializer.dims:
        dims.add().dim_value = dim
    return type


def get_constant_node_value(node: onnx.NodeProto, name: str) -> onnx.TensorProto | None:
    if (
        node.op_type != "Constant"
        or node.domain not in {"", "ai.onnx"}
        or len(node.attribute) != 1
    ):
        return None
    attr = node.attribute[0]
    if attr.ref_attr_name:
        return None
    attr_name = attr.name
    value = onnx.helper.get_attribute_value(attr)

    if isinstance(value, onnx.TensorProto):
        # Two names exist in this case: we use tensorproto as is (with original name)
        return value
    shape: list[int]
    if attr_name == "value_int":
        dtype = onnx.TensorProto.INT64
        shape = []
        value = [value]
    elif attr_name == "value_float":
        dtype = onnx.TensorProto.FLOAT
        shape = []
        value = [value]
    elif attr_name == "value_string":
        dtype = onnx.TensorProto.STRING
        shape = []
        value = [value]
    elif attr_name == "value_ints":
        dtype = onnx.TensorProto.INT64
        shape = [len(value)]
    elif attr_name == "value_floats":
        dtype = onnx.TensorProto.FLOAT
        shape = [len(value)]
    elif attr_name == "value_strings":
        dtype = onnx.TensorProto.STRING
        shape = [len(value)]
    else:
        return None  # sparse tensors not handled
    return onnx.helper.make_tensor(name, dtype, shape, value)
