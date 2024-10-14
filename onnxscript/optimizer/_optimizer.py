# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

from onnxscript import ir, rewriter
from onnxscript.optimizer import _constant_folding, _inliner
from onnxscript.optimizer._remove_unused import remove_unused_nodes
from onnxscript.rewriter import (
    broadcast_to_matmul,
    cast_constant_of_shape,
    gemm_to_matmul_add,
    no_op,
)

logger = logging.getLogger(__name__)

_DEFAULT_REWRITE_RULES = [
    *no_op.rules.rules,  # TODO: merge this rule into constant folding?
    *broadcast_to_matmul.rules.rules,
    gemm_to_matmul_add.rule,
    *cast_constant_of_shape.rules.rules,
]


def optimize_ir(
    model: ir.Model,
    num_iterations: int = 2,
    *,
    onnx_shape_inference: bool = True,
    stop_if_no_change: bool = True,
    input_size_limit: int = _constant_folding.DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT,
    output_size_limit: int = _constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT,
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
        stop_if_no_change: Not supported currently (has no effect). Meant to stop the
            outer optimization loop if no change is detected in one iteration.
    """
    del stop_if_no_change  # Looks like rewriter doesn't support this yet.
    _inliner.inline(model)
    for _ in range(num_iterations):
        _constant_folding.fold_constants(
            model,
            onnx_shape_inference=onnx_shape_inference,
            input_size_limit=input_size_limit,
            output_size_limit=output_size_limit,
        )
        rewriter.rewrite(model, pattern_rewrite_rules=_DEFAULT_REWRITE_RULES)
    remove_unused_nodes(model)
