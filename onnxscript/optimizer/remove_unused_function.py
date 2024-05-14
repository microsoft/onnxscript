# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging

import onnx

from onnxscript import ir

logger = logging.getLogger(__name__)


class UnusedFunctionRemover(ir.passes.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.used: set[ir.OperatorIdentifier] = set()

    def _call_function(self, function: ir.Function) -> None:
        if function.identifier() in self.used:
            # The function and its nodes are already recorded as used
            return
        self.used.add(function.identifier())
        for node in function:
            self.call_node_recursive(node)

    def call_node(self, node: ir.Node) -> None:
        op_identifier = node.op_identifier()
        self.used.add(op_identifier)
        if op_identifier in self.model.functions:
            self._call_function(self.model.functions[op_identifier])

    def exit_pass(self) -> None:
        # Update the model to remove unused functions
        count = 0
        for op_identifier in list(self.model.functions):
            if op_identifier not in self.used:
                del self.model.functions[op_identifier]
                count += 1
        logger.info("Removed %s unused function protos", count)
        logger.debug("Functions left: %s", [f.name for f in self.model.functions])


def remove_unused_functions(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    """Removes unused function protos from the model."""
    # TODO(justinchuby): Update this to accept an ir.Model
    model = ir.serde.deserialize_model(model_proto)
    UnusedFunctionRemover()(model)
    model_proto = ir.serde.serialize_model(model)

    return model_proto
