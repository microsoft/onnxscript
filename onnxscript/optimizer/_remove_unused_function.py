# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import TypeVar

import onnx

from onnxscript import ir

logger = logging.getLogger(__name__)


TModel = TypeVar("TModel", ir.Model, onnx.ModelProto)


def _clean_up_unused_functions(model: ir.Model, unused: set[ir.OperatorIdentifier]) -> None:
    """Removes unused functions from the model."""
    for op_identifier in unused:
        del model.functions[op_identifier]

    logger.info("Removed %s unused functions", len(unused))
    logger.debug("Functions left: %s", list(model.functions))
    logger.debug("Functions removed: %s", unused)


class RemoveUnusedFunctionPass(ir.passes.PassBase):
    def __init__(self):
        super().__init__()
        self.used: set[ir.OperatorIdentifier] | None = None

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        self.used = set()
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            self._call_node(model, node)

        # Update the model to remove unused functions
        unused = set(model.functions) - self.used
        if not unused:
            logger.info("No unused functions to remove")
            return ir.passes.PassResult(model, modified=False)

        _clean_up_unused_functions(model, unused)
        self.used = None
        return ir.passes.PassResult(model, modified=True)

    def _call_function(self, model: ir.Model, function: ir.Function) -> None:
        assert self.used is not None
        if function.identifier() in self.used:
            # The function and its nodes are already recorded as used
            return
        self.used.add(function.identifier())
        for node in ir.traversal.RecursiveGraphIterator(function):
            self._call_node(model, node)

    def _call_node(self, model: ir.Model, node: ir.Node) -> None:
        op_identifier = node.op_identifier()
        if op_identifier not in model.functions:
            return
        self._call_function(model, model.functions[op_identifier])


def remove_unused_functions(model: TModel) -> TModel:
    """Removes unused function protos from the model."""

    if isinstance(model, ir.Model):
        return RemoveUnusedFunctionPass()(model).model  # type: ignore[return-value]

    model_ = ir.serde.deserialize_model(model)
    result = RemoveUnusedFunctionPass()(model_)
    return ir.serde.serialize_model(result.model)
