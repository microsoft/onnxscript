# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx_ir as ir
import onnx_ir.passes.common as common_passes

from onnxscript import rewriter
from onnxscript.optimizer import _constant_folding

logger = logging.getLogger(__name__)


def optimize_ir(
    model: ir.Model,
    num_iterations: int = 2,
    *,
    onnx_shape_inference: bool = True,
    stop_if_no_change: bool = True,
    input_size_limit: int = _constant_folding.DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
    output_size_limit: int = _constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT,
    inline: bool = True,
) -> None:
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
    """
    passes = [
        ir.passes.PassManager(
            [
                _constant_folding.FoldConstantsPass(
                    shape_inference=onnx_shape_inference,
                    input_size_limit=input_size_limit,
                    output_size_limit=output_size_limit,
                ),
                rewriter.RewritePass(rewriter._DEFAULT_REWRITE_RULES),
                common_passes.RemoveUnusedNodesPass(),
                common_passes.RemoveUnusedFunctionsPass(),
                common_passes.RemoveUnusedOpsetsPass(),
            ],
            steps=num_iterations,
            early_stop=stop_if_no_change,
        ),
        common_passes.RemoveUnusedNodesPass(),
        common_passes.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0),
        common_passes.LiftSubgraphInitializersToMainGraphPass(),
        common_passes.DeduplicateInitializersPass(),
        common_passes.CommonSubexpressionEliminationPass(),
    ]
    if inline:
        # Inline all functions first before optimizing
        passes = [common_passes.InlinePass(), *passes]
    optimizer_pass = ir.passes.Sequential(*passes)
    assert optimizer_pass.in_place
    result = optimizer_pass(model)
    assert result.model is model
