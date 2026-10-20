# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule
from onnxscript.rewriter.rules.common._broadcast_to_matmul import (
    _compute_matmul_output_shape,
    check_if_not_need_reshape,
)

logger = logging.getLogger(__name__)


def _compute_broadcast_shape(
    shape1: Sequence[int],
    shape2: Sequence[int],
) -> list[int] | None:
    """Compute multidirectional broadcast shape of two static shapes."""
    try:
        return list(np.broadcast_shapes(tuple(shape1), tuple(shape2)))
    except ValueError:
        return None


# Pattern to match against
def reshape_gemm_reshape_pattern(op, input_a, input_b, input_c, shape_a, shape_c):
    reshape_a = op.Reshape(input_a, shape_a)
    # TODO: Temporary workaround to support benchmodels.
    # Tracked by https://github.com/microsoft/onnx-rewriter/issues/197.
    gemm = op.Gemm(reshape_a, input_b, input_c, alpha=1.0, beta=1.0)
    return op.Reshape(gemm, shape_c)


def matmul_add(op, input_a, input_b, input_c, **_):
    matmul = op.MatMul(input_a, input_b)
    return op.Add(matmul, input_c)


def check_if_not_need_reshape_for_gemm(context, *args, **kwargs) -> bool:
    """Reject Gemm rewrites if transposes exist or replacement output shape changes."""
    final_reshape = context.root if context and hasattr(context, "root") else None
    if final_reshape is None:
        return False
    gemm = final_reshape.inputs[0].producer()
    if gemm is None or gemm.op_type != "Gemm":
        return False
    if gemm.attributes.get_int("transA", 0) != 0 or gemm.attributes.get_int("transB", 0) != 0:
        logger.info("Gemm transpose attributes are not preserved by MatMul replacement.")
        return False

    input_c = kwargs.get("input_c")
    if input_c is None and len(args) > 2:
        input_c = args[2]
    if input_c is None and len(gemm.inputs) > 2:
        input_c = gemm.inputs[2]
    if input_c is None or input_c.shape is None:
        logger.info("The shape of 'input_c' is not statically known.")
        return False
    if any(isinstance(dim, ir.SymbolicDim) for dim in input_c.shape):
        logger.info("Symbolic dimensions in 'input_c' are not yet supported.")
        return False
    try:
        c_shape = list(input_c.shape.numpy())
    except (TypeError, ValueError):
        return False

    if not check_if_not_need_reshape(context, *args, **kwargs):
        return False

    input_a = (
        kwargs.get("input_a") if "input_a" in kwargs else (args[0] if len(args) > 0 else None)
    )
    input_b = (
        kwargs.get("input_b") if "input_b" in kwargs else (args[1] if len(args) > 1 else None)
    )
    if input_a is None:
        reshape_a = gemm.inputs[0].producer()
        if reshape_a is not None and reshape_a.inputs:
            input_a = reshape_a.inputs[0]
    if input_b is None and len(gemm.inputs) > 1:
        input_b = gemm.inputs[1]
    if input_a is None or input_b is None:
        return False

    matmul_output_shape = _compute_matmul_output_shape(
        list(input_a.shape.numpy()), list(input_b.shape.numpy())
    )
    if matmul_output_shape is None:
        return False

    replacement_add_shape = _compute_broadcast_shape(matmul_output_shape, c_shape)
    if replacement_add_shape is None:
        logger.info(
            "MatMul output shape %s and input_c shape %s are not broadcastable.",
            matmul_output_shape,
            c_shape,
        )
        return False

    # Establish complete replacement Add(MatMul(X, Y), C) output shape.
    # Reject the rewrite if bias broadcasting expands or alters the expected final output shape.
    if replacement_add_shape != matmul_output_shape:
        logger.info(
            "Replacement Add output shape %s differs from expected output shape %s due to bias broadcasting.",
            replacement_add_shape,
            matmul_output_shape,
        )
        return False

    return True


gemm_to_matmul_add_rule = RewriteRule(
    reshape_gemm_reshape_pattern, matmul_add, check_if_not_need_reshape_for_gemm
)
