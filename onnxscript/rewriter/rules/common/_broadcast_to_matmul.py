# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import math
from typing import Sequence

import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)


def _get_allowzero(reshape: ir.Node | None) -> int:
    """Return the allowzero attribute from an exact matched Reshape node."""
    if reshape is not None and reshape.op_type == "Reshape":
        attr = reshape.attributes.get("allowzero")
        if attr is not None:
            return int(getattr(attr, "value", attr))
    return 0


def _resolve_target_shape(
    target_shape: Sequence[int],
    input_shape: Sequence[int],
    allowzero: int = 0,
) -> list[int] | None:
    """Resolve target reshape dimension values using input dimensions.

    When allowzero=0 (default ONNX Reshape semantics), a 0 in target_shape
    copies the input dimension at the corresponding index.
    When allowzero=1, a 0 in target_shape is preserved as literal dimension 0.
    """
    resolved: list[int] = []
    inferred_idx = -1
    for idx, dim in enumerate(target_shape):
        if dim == 0:
            if allowzero == 1:
                resolved.append(0)
            elif idx < len(input_shape):
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
    elif math.prod(resolved) != math.prod(input_shape):
        return None

    return resolved


def _validate_reshape_layout(
    input_shape: Sequence[int],
    target_shape: Sequence[int],
    is_input_b: bool = False,
    allowzero: int = 0,
) -> bool:
    """Validate that removing a Reshape preserves matrix computation and batch-axis semantics.

    To safely eliminate a Reshape prior to MatMul:
    1. Matrix computation dimensions (contracting dimension and row/column dimension)
       must match between the input and the reshaped tensor (taking into account 1-D expansion).
    2. Batch-axis layout must be preserved: dimensions cannot be transposed, permuted,
       or altered beyond insertion or removal of leading size-1 dimensions.
    """
    resolved_target = _resolve_target_shape(target_shape, input_shape, allowzero=allowzero)
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


def _compute_matmul_output_shape(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
) -> list[int] | None:
    """Compute the output shape of MatMul(input_a, input_b) following ONNX MatMul semantics.

    Returns None if shapes are not broadcastable or contracting dimensions mismatch.
    """
    a_shape = list(input_a_shape)
    b_shape = list(input_b_shape)
    a_rank = len(a_shape)
    b_rank = len(b_shape)

    if a_rank < 1 or b_rank < 1:
        return None

    mimic_matmul_broadcast_behavior_a = False
    mimic_matmul_broadcast_behavior_b = False

    # 1.a. If the first input is 1-D, check whether
    # the dim matches the last second dim of the second input.
    if a_rank < 2:
        if b_rank < 2:
            logger.info("Optimization of dot product is not supported yet.")
            return None
        if a_shape[-1] != b_shape[-2]:
            logger.info("Original shape is not MatMul compatible.")
            return None
        a_shape = [1, *a_shape]
        a_rank = len(a_shape)
        mimic_matmul_broadcast_behavior_a = True

    # 1.b. If the second input is 1-D, check whether
    # the dim matches the last dim of the first input.
    if b_rank < 2:
        if b_shape[-1] != a_shape[-1]:
            logger.info("Original shape is not MatMul compatible.")
            return None
        b_shape = [*b_shape, 1]
        b_rank = len(b_shape)
        mimic_matmul_broadcast_behavior_b = True

    # 1.c. Contracting dimensions check
    if a_shape[-1] != b_shape[-2]:
        logger.info("Original contracting dimensions do not match.")
        return None

    # 1.d. Batch dimensions broadcasting
    batch_a = a_shape[:-2]
    batch_b = b_shape[:-2]
    broadcast_batch: list[int] = []

    for dim_from_a, dim_from_b in zip(reversed(batch_a), reversed(batch_b)):
        if dim_from_a != dim_from_b and dim_from_a != 1 and dim_from_b != 1:
            logger.info("Original shape is not broadcastable.")
            return None
        broadcast_dim = dim_from_b if dim_from_a == 1 else dim_from_a
        broadcast_batch.insert(0, broadcast_dim)

    # Prepend remaining leading batch dimensions from longer shape
    if len(batch_a) > len(batch_b):
        broadcast_batch = batch_a[: len(batch_a) - len(batch_b)] + broadcast_batch
    elif len(batch_b) > len(batch_a):
        broadcast_batch = batch_b[: len(batch_b) - len(batch_a)] + broadcast_batch

    out_shape = [*broadcast_batch, a_shape[-2], b_shape[-1]]

    if mimic_matmul_broadcast_behavior_b and b_rank == 2 and b_shape[-1] == 1:
        out_shape.pop(-1)
    if mimic_matmul_broadcast_behavior_a and a_rank == 2 and a_shape[0] == 1:
        out_shape.pop(
            -2
            if not (mimic_matmul_broadcast_behavior_b and b_rank == 2 and b_shape[-1] == 1)
            else -1
        )

    return out_shape


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

    final_reshape = context.root if context and hasattr(context, "root") else None
    matched_node = final_reshape.inputs[0].producer() if final_reshape is not None else None
    if matched_node is None:
        return False

    # Read attributes from the matched Reshape nodes, never from other consumers of shape values.
    matched_reshape_a = matched_node.inputs[0].producer() if shape_a is not None else None
    matched_reshape_b = matched_node.inputs[1].producer() if shape_b is not None else None
    allowzero_a = _get_allowzero(matched_reshape_a)
    if shape_a is not None:
        shape_a_tensor = shape_a.const_value
        if shape_a_tensor is None or len(shape_a_tensor.shape) != 1:
            logger.info("The value 'shape_a' is not statically known or 1-D.")
            return False
        if not _validate_reshape_layout(
            input_a_shape,
            shape_a_tensor.numpy().tolist(),
            is_input_b=False,
            allowzero=allowzero_a,
        ):
            return False

    allowzero_b = _get_allowzero(matched_reshape_b)
    if shape_b is not None:
        shape_b_tensor = shape_b.const_value
        if shape_b_tensor is None or len(shape_b_tensor.shape) != 1:
            logger.info("The value 'shape_b' is not statically known or 1-D.")
            return False
        if not _validate_reshape_layout(
            input_b_shape,
            shape_b_tensor.numpy().tolist(),
            is_input_b=True,
            allowzero=allowzero_b,
        ):
            return False

    # Compute matched MatMul output shape in the original graph
    matched_a_shape = (
        _resolve_target_shape(
            shape_a.const_value.numpy().tolist(), input_a_shape, allowzero=allowzero_a
        )
        if shape_a is not None
        else input_a_shape
    )
    matched_b_shape = (
        _resolve_target_shape(
            shape_b.const_value.numpy().tolist(), input_b_shape, allowzero=allowzero_b
        )
        if shape_b is not None
        else input_b_shape
    )
    if matched_a_shape is None or matched_b_shape is None:
        return False

    matched_matmul_shape = _compute_matmul_output_shape(matched_a_shape, matched_b_shape)
    if matched_matmul_shape is None:
        logger.info("Matched MatMul inputs are not valid for matrix multiplication.")
        return False

    # Resolve shape_c against matched MatMul output, honoring final Reshape's allowzero attribute
    allowzero_c = _get_allowzero(final_reshape)
    resolved_final_output_shape = _resolve_target_shape(
        shape_c_tensor.numpy().tolist(), matched_matmul_shape, allowzero=allowzero_c
    )
    if resolved_final_output_shape is None:
        logger.info("Target shape_c cannot be resolved against matched MatMul output shape.")
        return False

    # Compute replacement MatMul output shape
    replacement_output_shape = _compute_matmul_output_shape(input_a_shape, input_b_shape)
    if replacement_output_shape is None:
        logger.info("Input shapes are not MatMul compatible for the replacement.")
        return False

    # Check if final output shape matches replacement output shape
    if resolved_final_output_shape != replacement_output_shape:
        logger.info(
            "Final output shape is not the same. Expected %s vs actual %s",
            resolved_final_output_shape,
            replacement_output_shape,
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
