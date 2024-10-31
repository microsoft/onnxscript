# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    # Functions
    "convert_version"
]

from onnxscript import ir
from onnxscript.version_converter._adapter_lib import pick_adapter_set

BASE_OPSET_VERSION = 18

def convert_version(model: ir.Model, target_version: int) -> ir.Model:
    """Convert the model to the specified ONNX opset version."""

    model_version = model.opset_imports.get("")
    if model_version == target_version:
        # No conversion needed
        return model

    # Iterate through all nodes in the graph
    # Set node.version as the model_version
    # TODO: Do this internally in IR?
    graph = model.graph
    for node in graph._nodes:
        node.version = model_version

    # Iterate from current model version -> target version
    # Updating each node based on the correct adapter [ver->ver+1]
    for opset_version in range(model_version, target_version):
        for node in graph._nodes:
            adapter_set = pick_adapter_set(opset_version)
            if node.op_type in adapter_set:
                adapter = adapter_set[node.op_type]
                # Call the correct adapter [ver->ver+1]
                adapter(node, opset_version + 1)

    return model
