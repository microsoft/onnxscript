# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript import ir
from onnxscript.version_converter._adapter_lib import _Adapter


class _VersionConverter:
    def __init__(self, model: ir.Model) -> None:
        self._functions = model.functions
        self.model_version = model.opset_imports.get("")

    def model_version_convert(self, model: ir.Model, target_version: int) -> None:
        if self.model_version == target_version:
            # No conversion needed
            return

        # Iterate through all nodes in the graph
        # Set node.version as the model_version
        # TODO: Do this internally in IR?
        for node in model.graph:
            node.version = self.model_version

        # Iterate from current model version -> target version
        # Updating each node based on the correct adapter [ver->ver+1]
        for opset_version in range(self.model_version, target_version):
            op_adapter = _Adapter(opset_version, opset_version + 1)
            op_adapter.visit_model(model)


def version_convert(model: ir.Model, target_version: int) -> None:
    """Convert the model to the specified ONNX opset version."""
    version_converter = _VersionConverter(model)
    version_converter.model_version_convert(model, target_version)
