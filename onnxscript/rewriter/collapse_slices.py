# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import sys

from onnxscript import ir
from onnxscript.rewriter import pattern

logger = logging.getLogger(__name__)


def check_if_redundant_slice(
    context,
    data: ir.Value,
    starts: ir.Value,
    ends: ir.Value,
    axes: ir.Value,
    steps: ir.Value,
    **_,
):
    """If the starts is 0, and the ends is equal to or grater than the shape of the specified axis, then the slice is redundant."""
    del context  # Reserved for future extensions

    starts = starts.const_value
    ends = ends.const_value
    axes = axes.const_value
    steps = steps.const_value

    if starts is None or ends is None or axes is None or steps is None:
        logger.info("The value 'start', 'end', 'axis', 'step' is not statically known.")
        return False

    if steps.numpy().item() != 1:
        return False

    if starts.numpy().item() != 0:
        return False

    if ends.numpy().item() == sys.maxsize:
        return True

    if data.shape is None:
        logger.info("The value 'data' shape is not statically known.")
        return False

    if ends.numpy().item() < data.shape[axes.numpy().item()]:
        return False

    return True


def _identity(op, data, **_):
    return op.Identity(data)


def _potential_redundant_slice(op, data, starts, ends, axes, steps):
    return op.Slice(data, starts, ends, axes, steps)


def _slice(op, data_a, starts_a, ends_a, axes_a, steps_a, starts_b, ends_b, axes_b, steps_b):
    new_axes = op.Concat(axes_a, axes_b, axis=0)
    return op.Slice(data_a, starts_a, ends_a, new_axes, steps_a)


def _two_consecutive_slices_pattern(
    op, data_a, starts_a, ends_a, axes_a, steps_a, starts_b, ends_b, axes_b, steps_b
):
    intermediate = op.Slice(data_a, starts_a, ends_a, axes_a, steps_a)
    return op.Slice(intermediate, starts_b, ends_b, axes_b, steps_b)


def check_if_slices_are_collpasable(
    context,
    starts_a: ir.Value,
    ends_a: ir.Value,
    steps_a: ir.Value,
    starts_b: ir.Value,
    ends_b: ir.Value,
    steps_b: ir.Value,
    **_,
):
    """

    In PyTorch axes (dim) is only allowed to be scalar, but in ONNX, op.Slice accpets a list of axes.
    Therefore, we collapse the axesof two op.Slices into a single list, when we find start, end,
    and step are the same.

    """
    del context  # Reserved for future extensions

    # cehck starts, ends, and steps are constants
    starts_a = starts_a.const_value
    ends_a = ends_a.const_value
    steps_a = steps_a.const_value
    starts_b = starts_b.const_value
    ends_b = ends_b.const_value
    steps_b = steps_b.const_value

    if starts_a is None or ends_a is None or steps_a is None:
        logger.info("The value 'start', 'end', 'step' is not statically known.")
        return False
    if starts_b is None or ends_b is None or steps_b is None:
        logger.info("The value 'start', 'end', 'step' is not statically known.")
        return False

    # check starts, ends, and steps are the same
    if starts_a.numpy() != starts_b.numpy():
        return False
    if ends_a.numpy() != ends_b.numpy():
        return False
    if steps_a.numpy() != steps_b.numpy():
        return False
    return True


# Register the rewrite rules
remove_redundant_slice = pattern.RewriteRule(
    _potential_redundant_slice,
    _identity,
    check_if_redundant_slice,
)

collapse_slices_rule = pattern.RewriteRule(
    _two_consecutive_slices_pattern,
    _slice,
    # We can use the same check_if_not_need_reshape function for both the rules,
    # as one_reshape_matmul_reshape_pattern is a subset of _two_reshapes_matmul_reshape_pattern.
    check_if_slices_are_collpasable,
)

# NOTE: The order of the rules is important. Larger pattern should be checked first.
rules = pattern.RewriteRuleSet([remove_redundant_slice])
