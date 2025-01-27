# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

from onnxscript import ir
from onnxscript.rewriter import pattern

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


def _identity_to_updates(op, data, indices, updates, **_):
    """Return the updates as the output.

    This is used when the ScatterND is redundant in terms of
    updating the whole data with the updates.

    """
    return op.Identity(updates)


def _potential_redundant_slice(op, data, starts, ends, axes, steps):
    """To identify a slice op"""
    return op.Slice(data, starts, ends, axes, steps)


def _potential_redundant_scatternd(op, data, indices, updates):
    """To identify a ScatterND op"""
    return op.ScatterND(data, indices, updates)


def _check_if_redundant_scatternd(
    context,
    data: ir.Value,
    indices: ir.Value,
    updates: ir.Value,
    **_,
):
    """If the indices is the same length as the first dim of data, and the shape of updates is equal to data, we can simply swap the whole value."""
    del context  # Reserved for future extensions

    # To validate data can be replaced directly by updates, we need to check the following:
    # 1. they have the same shape
    if data.shape is None:
        logger.info("The value 'data' shape is not statically known.")
        return False
    if updates.shape is None:
        logger.info("The value 'updates' shape is not statically known.")
        return False
    if data.shape != updates.shape:
        logger.info("The shape of 'data' and 'updates' are different.")
        return False

    # 2. the indices is referring to the whole data, which is from 0 to data.shape[0]
    if indices.const_value is None:
        logger.info("The value 'indices' is not statically known.")
        return False
    if indices.const_value.numpy().tolist() != [[i] for i in range(data.shape[0])]:  # type: ignore[arg-type]
        logger.info("The 'indices' is not referring to the whole data.")
        return False

    return True


# Register the rewrite rules
remove_redundant_slice = pattern.RewriteRule(
    _potential_redundant_slice,
    _identity_to_itself,
    _check_if_redundant_slice,
)

remove_redundant_scatternd = pattern.RewriteRule(
    _potential_redundant_scatternd,
    _identity_to_updates,
    _check_if_redundant_scatternd,
)

# NOTE: The order of the rules is important. Larger pattern should be checked first.
rules = pattern.RewriteRuleSet([remove_redundant_slice, remove_redundant_scatternd])
