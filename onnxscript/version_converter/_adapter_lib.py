# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stores all operator schema related changes between opsets"""

from __future__ import annotations

from onnxscript import ir

### Adapters


# Adapter to convert from an attribute to a node input
def adapter_attr_to_input(attr_name: str):
    def adapt(node: ir.Node):
        assert attr_name in node._attributes
        _attr = node._attributes.pop(attr_name)
        # Create an ir.Value obj with properties of the axis attribute
        attr_as_input = ir.Value(
            name=_attr.name,
        )
        # Add the ir.Value as inputs to the node and graph
        node._inputs = node._inputs + (attr_as_input,)
        node.graph._inputs = node.graph._inputs + [attr_as_input]

    return adapt


### Op specific adapters


# Adapter for GridSample 19 -> 20
def adapter_gridsample_19_20(node: ir.Node):
    for attr in node._attributes:
        if attr == "mode":
            mode_value = node._attributes[attr].value
            if mode_value == "bilinear":
                node._attributes[attr].value = "linear"
            elif mode_value == "bicubic":
                node._attributes[attr].value = "cubic"


# Adapter for Group 19 -> 20
def adapter_groupnormalization_20_21(node: ir.Node):
    pass


_ADAPTERS_18_19 = {
    "Equal": None,
    "AveragePool": None,
    "Cast": None,
    "CastLike": None,
    "Constant": None,
    "DequantizeLinear": None,
    "Identity": None,
    "If": None,
    "Loop": None,
    "Pad": None,
    "QuantizeLinear": None,
    "Reshape": None,
    "Scan": None,
    "Shape": None,
    "Size": None,
}


_ADAPTERS_19_20 = {
    "DFT": adapter_attr_to_input("axis"),
    "ConstantOfShape": None,
    "IsInf": None,
    "IsNan": None,
    "ReduceMax": None,
    "ReduceMin": None,
    "GridSample": adapter_gridsample_19_20,
}


_ADAPTERS_20_21 = {
    "Cast": None,
    "CastLike": None,
    "Constant": None,
    "ConstantOfShape": None,
    "DequantizeLinear": None,
    "Flatten": None,
    "GroupNormalization": adapter_groupnormalization_20_21,
    "Identity": None,
    "If": None,
    "Loop": None,
    "Pad": None,
    "QLinearMatmul": None,
    "QuantizeLinear": None,
    "Reshape": None,
    "Scan": None,
    "Shape": None,
    "Size": None,
    "Squeeze": None,
    "Transpose": None,
    "Unsqueeze": None,
}


_ADAPTER_SETS = {
    18: _ADAPTERS_18_19,
    19: _ADAPTERS_19_20,
    20: _ADAPTERS_20_21,
}


def pick_adapter_set(current_version: int):
    return _ADAPTER_SETS[current_version]
