# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stores all operator schema related changes between opsets"""

from onnxscript.version_converter.version_converter import (
    adapter_compatible,
    adapter_axis_attr_to_input,
    adapter_gridsample_19_20,
    adapter_groupnormalization_20_21,
)


_ADAPTERS_18_19 = {
    "Equal": adapter_compatible,
    "AveragePool": adapter_compatible,
    "Cast": adapter_compatible,
    "CastLike": adapter_compatible,
    "Constant": adapter_compatible,
    "DequantizeLinear": adapter_compatible,
    "Identity": adapter_compatible,
    "If": adapter_compatible,
    "Loop": adapter_compatible,
    "Pad": adapter_compatible,
    "QuantizeLinear": adapter_compatible,
    "Reshape": adapter_compatible,
    "Scan": adapter_compatible,
    "Shape": adapter_compatible,
    "Size": adapter_compatible,
}


_ADAPTERS_19_20 = {
    "DFT": adapter_axis_attr_to_input,
    "ConstantOfShape": adapter_compatible,
    "IsInf": adapter_compatible,
    "IsNan": adapter_compatible,
    "ReduceMax": adapter_compatible,
    "ReduceMin": adapter_compatible,
    "GridSample": adapter_gridsample_19_20,
}


_ADAPTERS_20_21 = {
    "Cast": adapter_compatible,
    "CastLike": adapter_compatible,
    "Constant": adapter_compatible,
    "ConstantOfShape": adapter_compatible,
    "DequantizeLinear": adapter_compatible,
    "Flatten": adapter_compatible,
    "GroupNormalization": [adapter_compatible, adapter_groupnormalization_20_21],
    "Identity": adapter_compatible,
    "If": adapter_compatible,
    "Loop": adapter_compatible,
    "Pad": adapter_compatible,
    "QLinearMatmul": adapter_compatible,
    "QuantizeLinear": adapter_compatible,
    "Reshape": adapter_compatible,
    "Scan": adapter_compatible,
    "Shape": adapter_compatible,
    "Size": adapter_compatible,
    "Squeeze": adapter_compatible,
    "Transpose": adapter_compatible,
    "Unsqueeze": adapter_compatible,
}


_ADAPTER_SETS = {
    18: _ADAPTERS_18_19,
    19: _ADAPTERS_19_20,
    20: _ADAPTERS_20_21,
}


def pick_adapter_set(current_version: int):
    return _ADAPTER_SETS[current_version]
