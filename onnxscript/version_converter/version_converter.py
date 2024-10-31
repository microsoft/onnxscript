# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript import ir

### Adapters

# Compatibility Adapter
def adapter_compatible(op: ir.Node, target_opset):
    op.version = target_opset


# Adapter to convert axis from an attribute to a node input
def adapter_axis_attr_to_input(op: ir.Node, target_opset):
    op.version = target_opset
    assert 'axis' in op._attributes
    axis_attr = op._attributes.pop("axis")
    # Create an ir.Value obj with properties of the axis attribute
    axis_input = ir.Value(
        name=axis_attr.name,
    )
    # Add the ir.Value as inputs to the node and graph
    op._inputs = op._inputs + (axis_input, )
    op.graph._inputs = op.graph._inputs + [axis_input]


### Op specific adapters

# Adapter for GridSample 19 -> 20
def adapter_gridsample_19_20(op: ir.Node, target_opset):
    op.version = target_opset
    for attr in op._attributes:
        if attr == 'mode':
            mode_value = op._attributes[attr].value
            if mode_value == 'bilinear':
                op._attributes[attr].value = 'linear'
            elif mode_value == 'bicubic':
                op._attributes[attr].value = 'cubic'


# Adapter for Group 19 -> 20
def adapter_groupnormalization_20_21(op: ir.Node, target_opset):
    op.version = target_opset
