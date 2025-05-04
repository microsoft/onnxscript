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

import onnxscript.ir.passes.common
import onnxscript.optimizer._constant_folding as constant_folding
from onnxscript import ir
from onnxscript.optimizer._constant_folding import (
    basic_constant_propagation,
)
from onnxscript.optimizer._constant_folding import (
    fold_constants as fold_constants_ir,
)
from onnxscript.optimizer._optimizer import optimize_ir

_ModelProtoOrIr = TypeVar("_ModelProtoOrIr", onnx.ModelProto, ir.Model)


def optimize(
    model: _ModelProtoOrIr,
    num_iterations: int = 2,
    *,
    onnx_shape_inference: bool = True,
    stop_if_no_change: bool = True,
    input_size_limit: int = constant_folding.DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
    output_size_limit: int = constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT,
    inline: bool = True,
) -> _ModelProtoOrIr:
    """Optimizes a model.

    Args:
        model: The model to be optimized.
        num_iterations: Number of times the optimization loop is repeated.
        onnx_shape_inference: Applies node-level shape-inference as part of optimization
        input_size_limit: Will not apply constant folding to ops with any input of size
            greater than this. Does not apply to special ops like Shape() and Size().
        output_size_limit: Will not rewrite any foldable-op into a Constant op if the size
            of the output tensor is greater than this.
        stop_if_no_change: Stop the optimization loop if no change is detected in an iteration.
        inline: If True, inlines all functions in the model.

    Returns:
        The optimized model. If the input was a ModelProto, the output will also be a
        ModelProto. If the input was an ir.Model, the output will also be an ir.Model.
    """
    if isinstance(model, ir.Model):
        # In this case, optimize is done inplace.
        # TODO(justinchuby): Maybe make functional
        optimize_ir(
            model,
            num_iterations=num_iterations,
            onnx_shape_inference=onnx_shape_inference,
            stop_if_no_change=stop_if_no_change,
            input_size_limit=input_size_limit,
            output_size_limit=output_size_limit,
            inline=inline,
        )
        return model
    else:
        assert isinstance(model, onnx.ModelProto)
        model_ir = ir.serde.deserialize_model(model)
        optimize_ir(
            model_ir,
            num_iterations=num_iterations,
            onnx_shape_inference=onnx_shape_inference,
            stop_if_no_change=stop_if_no_change,
            input_size_limit=input_size_limit,
            output_size_limit=output_size_limit,
            inline=inline,
        )
        # Move the model back to the proto
        new_proto = ir.serde.serialize_model(model_ir)
        return new_proto


def inline(model: ir.Model) -> None:
    """Inline all function calls (recursively) in the model."""
    if model.functions:
        onnxscript.ir.passes.common.InlinePass()(model)


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
        onnxscript.ir.passes.common.RemoveUnusedNodesPass()(model)
    else:
        model_ir = ir.serde.deserialize_model(model)
        model_ir = onnxscript.ir.passes.common.RemoveUnusedNodesPass()(model_ir).model
        new_proto = ir.serde.serialize_model(model_ir)
        model.Clear()
        model.CopyFrom(new_proto)


def remove_unused_functions(model: ir.Model | onnx.ModelProto) -> None:
    """Removes unused functions from a model inplace."""
    if isinstance(model, ir.Model):
        onnxscript.ir.passes.common.RemoveUnusedFunctionsPass()(model)
    else:
        model_ir = ir.serde.deserialize_model(model)
        model_ir = onnxscript.ir.passes.common.RemoveUnusedFunctionsPass()(model_ir).model
        new_proto = ir.serde.serialize_model(model_ir)
        model.Clear()
        model.CopyFrom(new_proto)
