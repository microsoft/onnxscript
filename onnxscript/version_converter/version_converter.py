# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript import ir
from onnxscript.version_converter._adapter_lib import pick_adapter_set


class _VersionConverter:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self.model_version = model.opset_imports.get("")
        self.custom_adapters = {}

    def _register_custom_adapters(self, custom_adapter_list) -> None:
        self.custom_adapters = custom_adapter_list

    def graph_version_convert(self, graph: ir.Graph, target_version: int) -> None:
        if self.model_version == target_version:
            # No conversion needed
            return

        # Iterate through all nodes in the graph
        # Set node.version as the model_version
        # TODO: Do this internally in IR?
        for node in graph:
            node.version = self.model_version

        # Iterate from current model version -> target version
        # Updating each node based on the correct adapter [ver->ver+1]
        for opset_version in range(self.model_version, target_version):
            for node in graph:
                adapter_set = pick_adapter_set(opset_version)
                if node.op_type in adapter_set:
                    adapter_func = adapter_set[node.op_type]
                    if adapter_func is not None:
                        adapter_func(node)
                    node.version = opset_version + 1
                # Apply custom adapters
                if (
                    self.custom_adapters is not None
                    and str(opset_version) in self.custom_adapters
                ):
                    custom_adapter_set = self.custom_adapters[str(opset_version)]
                    if node.op_type in custom_adapter_set:
                        adapter_func = custom_adapter_set[node.op_type]
                        if adapter_func is not None:
                            adapter_func(node)


def version_convert(model: ir.Model, target_version: int, custom_adapters=None) -> None:
    """Convert the model to the specified ONNX opset version."""
    version_converter = _VersionConverter(model)
    version_converter._register_custom_adapters(custom_adapters)
    version_converter.graph_version_convert(model.graph, target_version)
