# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import math
from typing import Sequence

import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)


def _resolve_target_shape(
    target_shape: Sequence[int], input_shape: Sequence[int]
) -> list[int] | None:
    """Resolve target reshape dimension values using input dimensions.

    Note: assumes default ONNX Reshape attribute allowzero=0,
    where a 0 in the shape tensor copies the input dimension at the corresponding index.
    """
    resolved: list[int] = []
    inferred_idx = -1
    for idx, dim in enumerate(target_shape):
        if dim == 0:
            if idx < len(input_shape):
                resolved.append(input_shape[idx])
            else:
                return None
        elif dim == -1:
            if inferred_idx != -1:
                return None
            inferred_idx = idx
            resolved.append(dim)
        elif dim > 0:
            resolved.append(dim)
        else:
            return None

    if inferred_idx != -1:
        total = math.prod(input_shape)
        known = math.prod(d for d in resolved if d != -1)
        if known == 0 or total % known != 0:
            return None
        resolved[inferred_idx] = total // known

    return resolved


def _validate_reshape_layout(
    input_shape: Sequence[int],
    target_shape: Sequence[int],
    is_input_b: bool = False,
) -> bool:
    """Validate that removing a Reshape preserves matrix computation and batch-axis semantics.

    To safely eliminate a Reshape prior to MatMul:
    1. Matrix computation dimensions (contracting dimension and row/column dimension)
       must match between the input and the reshaped tensor (taking into account 1-D expansion).
    2. Batch-axis layout must be preserved: dimensions cannot be transposed, permuted,
       or altered beyond insertion or removal of leading size-1 dimensions.
    """
    resolved_target = _resolve_target_shape(target_shape, input_shape)
    if resolved_target is None:
        return False

    orig = list(input_shape)
    target = list(resolved_target)

    # Normalize 1-D inputs to 2-D matrix representations per ONNX MatMul semantics
    if len(orig) < 2:
        orig = [*orig, 1] if is_input_b else [1, *orig]
    if len(target) < 2:
        target = [*target, 1] if is_input_b else [1, *target]

    # 1. Verify matrix computation dimensions match
    if orig[-2:] != target[-2:]:
        logger.info(
            "Matrix computation dimensions do not match: input %s vs reshape %s",
            orig[-2:],
            target[-2:],
        )
        return False

    # 2. Verify batch-axis semantics: batch dimensions must match when right-aligned
    # (i.e. only leading size-1 dimensions may differ)
    batch_orig = orig[:-2]
    batch_target = target[:-2]
    max_len = max(len(batch_orig), len(batch_target))
    padded_orig = [1] * (max_len - len(batch_orig)) + batch_orig
    padded_target = [1] * (max_len - len(batch_target)) + batch_target

    if padded_orig != padded_target:
        logger.info(
            "Batch-axis layout not preserved: input batch %s vs reshape batch %s",
            batch_orig,
            batch_target,
        )
        return False

    return True


def check_if_not_need_reshape(
    context,
    input_a: ir.Value,
    input_b: ir.Value,
    shape_c: ir.Value,
    shape_a: ir.Value | None = None,
    shape_b: ir.Value | None = None,
    **_,
) -> bool:
    """Condition to check if we need to replace the pattern.

    If matmul broadcasting is enough, then we don't need the reshapes.

    To validate this, we need to check the following:
    1. Input shapes check: input_a and input_b should be broadcastable
    2. Output shape check: shape_c should be the same as the output shape from the matmul(input_a, input_b)

    If the above are true, then we don't need the reshapes.

    Returns:
        True if we need to replace the pattern, False otherwise.
    """
    del context  # Reserved for future extensions

    input_a_shape = input_a.shape
    input_b_shape = input_b.shape
    shape_c_tensor = shape_c.const_value
    if shape_c_tensor is None:
        logger.info("The value 'shape_c' is not statically known.")
        return False

    if len(shape_c_tensor.shape) != 1:
        logger.info(
            "Unexpected final shape. The shape of 'shape' value is %s",
            shape_c_tensor.shape,
        )
        return False

    # NOTE: When there is a subset match with a pattern. The MatchResult won't have the shape
    # information. So, we need to check if the shape is None and return False.
    if input_a_shape is None or input_b_shape is None:
        logger.info("Shape information is not available for the inputs and outputs.")
        return False
    if any(isinstance(dim, ir.SymbolicDim) for dim in input_a_shape):
        logger.info("Symbolic dimensions are not yet supported.")
        return False
    if any(isinstance(dim, ir.SymbolicDim) for dim in input_b_shape):
        logger.info("Symbolic dimensions are not yet supported.")
        return False
    input_a_shape = input_a_shape.numpy()  # type: ignore[assignment]
    input_b_shape = input_b_shape.numpy()  # type: ignore[assignment]
    shape_c = shape_c_tensor.numpy().tolist()  # type: ignore[assignment]

    # Validate that removed reshapes preserve matrix computation and batch layout
    if shape_a is not None:
        shape_a_tensor = shape_a.const_value
        if shape_a_tensor is None or len(shape_a_tensor.shape) != 1:
            logger.info("The value 'shape_a' is not statically known or 1-D.")
            return False
        if not _validate_reshape_layout(
            input_a_shape, shape_a_tensor.numpy().tolist(), is_input_b=False
        ):
            return False

    if shape_b is not None:
        shape_b_tensor = shape_b.const_value
        if shape_b_tensor is None or len(shape_b_tensor.shape) != 1:
            logger.info("The value 'shape_b' is not statically known or 1-D.")
            return False
        if not _validate_reshape_layout(
            input_b_shape, shape_b_tensor.numpy().tolist(), is_input_b=True
        ):
            return False

    a_rank = len(input_a_shape)
    b_rank = len(input_b_shape)

    # 1. Check if input shapes are broadcastable
    # 1.a. If the first input is 1-D, check whether
    # the dim matches the last second dim of the second input.
    mimic_matmul_broadcast_behavior_a = False
    mimic_matmul_broadcast_behavior_b = False
    if a_rank < 2:
        if b_rank < 2:
            logger.info("Optimization of dot product is not supported yet.")
            return False
        if input_a_shape[-1] != input_b_shape[-2]:
            logger.info("Original shape is not MatMul compatible.")
            return False
        else:
            input_a_shape = [1, *input_a_shape]  # type: ignore[assignment]
            a_rank = len(input_a_shape)
            mimic_matmul_broadcast_behavior_a = True
    # 1.b. If the second input is 1-D, check whether
    # the dim matches the last dim of the first input.
    if b_rank < 2:
        if input_b_shape[-1] != input_a_shape[-1]:
            logger.info("Original shape is not MatMul compatible.")
            return False
        else:
            input_b_shape = [*input_b_shape, 1]  # type: ignore[assignment]
            b_rank = len(input_b_shape)
            mimic_matmul_broadcast_behavior_b = True
    # 1.c. If both inputs are at least 2-D, check whether
    # the last dimension of the first input matches the second
    # last dimension of the second input, and shape[:-2] are
    # broadcastable.
    input_a_shape_except_second_last_dim = [*input_a_shape[:-2], *[input_a_shape[-1]]]
    input_b_shape_except_last_dim = input_b_shape[:-1]
    broadcast_matmul_output_shape = [input_a_shape[-2], input_b_shape[-1]]
    for idx, (dim_from_a, dim_from_b) in enumerate(
        zip(
            reversed(input_a_shape_except_second_last_dim),
            reversed(input_b_shape_except_last_dim),
        )
    ):
        if idx == 0:
            if dim_from_a != dim_from_b:
                logger.info("Original contracting dimensions do not match.")
                return False
        else:
            if dim_from_a != dim_from_b and dim_from_a != 1 and dim_from_b != 1:
                logger.info("Original shape is not broadcastable.")
                return False
            broadcast_dim = dim_from_b if dim_from_a == 1 else dim_from_a
            broadcast_matmul_output_shape = [
                broadcast_dim,  # type: ignore[list-item]
                *broadcast_matmul_output_shape,
            ]

    # 2. Check if output shape is the same as the output shape from the matmul(input_a, input_b)
    # Prepend the broadcast_matmul_output_shape with the longer shape of input
    if a_rank > b_rank:
        longer_shape = input_a_shape
        shorter_shape = input_b_shape
    else:
        longer_shape = input_b_shape
        shorter_shape = input_a_shape
    broadcast_matmul_output_shape = [
        *longer_shape[: -len(shorter_shape)],
        *broadcast_matmul_output_shape,
    ]
    if mimic_matmul_broadcast_behavior_b and b_rank == 2 and input_b_shape[-1] == 1:
        # If input_b is expanded to 2-D, then we need to remove the last dimension
        broadcast_matmul_output_shape = broadcast_matmul_output_shape[:-1]
    if mimic_matmul_broadcast_behavior_a and a_rank == 2 and input_a_shape[0] == 1:
        # If input_a is expanded to 2-D, then we need to remove the first dimension
        # of input_a, which would be the -2nd dimension of the output shape.
        broadcast_matmul_output_shape.pop(-2)
    if shape_c != broadcast_matmul_output_shape:
        logger.info(
            "Final output shape is not the same. Expected %s vs actual %s",
            shape_c,
            broadcast_matmul_output_shape,
        )
        return False

    return True


def _two_reshapes_matmul_reshape_pattern(op, input_a, input_b, shape_a, shape_b, shape_c):
    # TODO: Modified from `value_ints` to `value` to match pattern in benchmark models.
    # This implementation misses pattern of Constants with `value_ints` attribute.
    # See more at https://github.com/microsoft/onnx-rewriter/issues/191.
    # A better solution is to improve pattern matching and avoid depending on writing
    # Constants in pattern. See https://github.com/microsoft/onnx-rewriter/issues/192.
    reshape_a = op.Reshape(input_a, shape_a)
    reshape_b = op.Reshape(input_b, shape_b)
    matmul = op.MatMul(reshape_a, reshape_b)
    return op.Reshape(matmul, shape_c)


def _matmul(op, input_a, input_b, **_):
    return op.MatMul(input_a, input_b)


def _one_reshape_matmul_reshape_pattern(op, input_a, input_b, shape_a, shape_c):
    reshape_a = op.Reshape(input_a, shape_a)
    matmul = op.MatMul(reshape_a, input_b)
    return op.Reshape(matmul, shape_c)


# Register the rewrite rules
two_reshapes_matmul_reshape_rule = RewriteRule(
    _two_reshapes_matmul_reshape_pattern,
    _matmul,
    check_if_not_need_reshape,
)
one_reshape_matmul_reshape_rule = RewriteRule(
    _one_reshape_matmul_reshape_pattern,
    _matmul,
    # We can use the same check_if_not_need_reshape function for both the rules,
    # as one_reshape_matmul_reshape_pattern is a subset of _two_reshapes_matmul_reshape_pattern.
    check_if_not_need_reshape,
)

# NOTE: The order of the rules is important. Larger pattern should be checked first.
rules = RewriteRuleSet([two_reshapes_matmul_reshape_rule, one_reshape_matmul_reshape_rule])
