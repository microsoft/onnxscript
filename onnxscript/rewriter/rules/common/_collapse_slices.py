# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

from onnxscript import ir
from onnxscript.rewriter._ir_utils import is_singleton_value
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)
_INT64_MAX = 9223372036854775807


def _check_if_redundant_slice(
    context,
    data: ir.Value,
    starts: ir.Value,
    ends: ir.Value,
    axes: ir.Value,
    steps: ir.Value,
    **_,
) -> bool:
    """If the starts is 0, and the ends is equal to or grater than the shape of the specified axis, then the slice is redundant."""
    del context  # Reserved for future extensions

    starts_const = starts.const_value
    ends_const = ends.const_value
    axes_const = axes.const_value
    steps_const = steps.const_value

    if starts_const is None or ends_const is None or axes_const is None or steps_const is None:
        logger.info("The value 'start', 'end', 'axis', 'step' is not statically known.")
        return False

    # Check if the values are scalar
    if starts_const.numpy().size != 1:  # type: ignore[union-attr]
        logger.info("The value 'start' is not a scalar.")
        return False
    if ends_const.numpy().size != 1:  # type: ignore[union-attr]
        logger.info("The value 'end' is not a scalar.")
        return False
    if axes_const.numpy().size != 1:  # type: ignore[union-attr]
        logger.info("The value 'axis' is not a scalar.")
        return False
    if steps_const.numpy().size != 1:  # type: ignore[union-attr]
        logger.info("The value 'step' is not a scalar.")
        return False

    if steps_const.numpy().item() != 1:
        logger.info("The value 'step' is not 1.")
        return False
    # starts is 0
    if starts_const.numpy().item() != 0:
        logger.info("The value 'start' is not 0.")
        return False
    # In case data.shape is not statically known, we still can tell the slice is redundant if ends is sys.maxsize
    if ends_const.numpy().item() == _INT64_MAX:
        return True
    if data.shape is None or data.shape.is_dynamic(axes_const.numpy().item()):
        logger.info("The value 'data' shape is not statically known.")
        return False
    if ends_const.numpy().item() < data.shape[axes_const.numpy().item()]:
        logger.info("The value 'end' is less than the shape of the specified axis.")
        return False

    return True


def _identity_to_itself(op, data, **_):
    """Return the input data as the output."""
    return op.Identity(data)


def _potential_redundant_slice(op, data, starts, ends, axes, steps):
    """To identify a slice op"""
    return op.Slice(data, starts, ends, axes, steps, _outputs=["slice_output"])


def _same_shape(op, data: ir.Value, slice_output: ir.Value, steps: ir.Value, **_):
    """Check if the shape of the slice output is the same as the data."""
    if data.shape is None or slice_output.shape is None:
        return False

    if not is_singleton_value(steps, 1):
        return False

    return data.shape == slice_output.shape


# Register the rewrite rules
collapse_slice_rule = RewriteRule(
    _potential_redundant_slice,
    _identity_to_itself,
    _check_if_redundant_slice,
)

collapse_slice2_rule = RewriteRule(
    _potential_redundant_slice,
    _identity_to_itself,
    _same_shape,
)

# NOTE: The second rule subsumes the first one. So, we may be able to remove the first one,
# provided shape-inference is run before the rewriter and computes the shape of the slice output.

rules = RewriteRuleSet([collapse_slice_rule, collapse_slice2_rule])
