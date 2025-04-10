# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import TypeVar

__all__ = [
    "basic_constant_propagation",
    "fold_constants_ir",
    "fold_constants",
    "inline",
    "optimize_ir",
    "optimize",
    "remove_unused_nodes",
]

import onnx

import onnxscript.ir.passes.common.unused_removal
import onnxscript.optimizer._constant_folding as constant_folding
from onnxscript import ir
from onnxscript.optimizer._constant_folding import (
    basic_constant_propagation,
)
from onnxscript.optimizer._constant_folding import (
    fold_constants as fold_constants_ir,
)
from onnxscript.optimizer._inliner import inline
from onnxscript.optimizer._optimizer import optimize_ir

_ModelProtoOrIr = TypeVar("_ModelProtoOrIr", onnx.ModelProto, ir.Model)


def optimize(model: _ModelProtoOrIr, *args, **kwargs) -> _ModelProtoOrIr:
    if isinstance(model, ir.Model):
        # In this case, optimize is done inplace.
        # TODO(justinchuby): Maybe make functional
        optimize_ir(model, *args, **kwargs)
        return model
    else:
        assert isinstance(model, onnx.ModelProto)
        model_ir = ir.serde.deserialize_model(model)
        optimize_ir(model_ir, *args, **kwargs)
        # Move the model back to the proto
        new_proto = ir.serde.serialize_model(model_ir)
        return new_proto


def fold_constants(
    model: ir.Model | onnx.ModelProto, *args, **kwargs
) -> constant_folding.FoldConstantsResult:
    """Fold constants in a model in place."""
    if isinstance(model, ir.Model):
        return constant_folding.fold_constants(model, *args, **kwargs)
    else:
        assert isinstance(model, onnx.ModelProto)
        model_proto = model
        model = ir.serde.deserialize_model(model_proto)
        result = constant_folding.fold_constants(model, *args, **kwargs)
        # Move the model back to the proto
        new_proto = ir.serde.serialize_model(model)
        model_proto.Clear()
        model_proto.CopyFrom(new_proto)
        return result


def remove_unused_nodes(model: ir.Model | onnx.ModelProto) -> None:
    """Removes unused nodes from a model inplace."""
    if isinstance(model, ir.Model):
        onnxscript.ir.passes.common.unused_removal.RemoveUnusedNodesPass()(model)
    else:
        model_ir = ir.serde.deserialize_model(model)
        model_ir = onnxscript.ir.passes.common.unused_removal.RemoveUnusedNodesPass()(
            model_ir
        ).model
        new_proto = ir.serde.serialize_model(model_ir)
        model.Clear()
        model.CopyFrom(new_proto)


def remove_unused_functions(model: ir.Model | onnx.ModelProto) -> None:
    """Removes unused functions from a model inplace."""
    if isinstance(model, ir.Model):
        onnxscript.ir.passes.common.unused_removal.RemoveUnusedFunctionsPass()(model)
    else:
        model_ir = ir.serde.deserialize_model(model)
        model_ir = onnxscript.ir.passes.common.unused_removal.RemoveUnusedFunctionsPass()(
            model_ir
        ).model
        new_proto = ir.serde.serialize_model(model_ir)
        model.Clear()
        model.CopyFrom(new_proto)
