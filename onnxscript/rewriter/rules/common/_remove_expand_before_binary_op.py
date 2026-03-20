# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fusion rule to remove an Expand node before a binary operator.

This implements the optimization:

    BinaryOp(Expand(x, shape), y) -> BinaryOp(x, y)
    BinaryOp(x, Expand(y, shape)) -> BinaryOp(x, y)

This is valid when the binary operator's broadcasting semantics would produce
the same output shape as first expanding the input and then applying the op.
"""

from __future__ import annotations

from onnxscript import ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import get_numpy_value
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

# Binary operators in ONNX standard opset that support numpy-style broadcasting.
_BROADCAST_BINARY_OPS: tuple[str, ...] = (
    "Add",
    "And",
    "BitShift",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "PRelu",
    "Sub",
    "Xor",
)


def _check_expand_removable(
    expand_input: ir.Value,
    shape: ir.Value,
    other_input: ir.Value,
) -> MatchResult:
    """Check if an Expand node can be safely removed before a binary op.

    The Expand node ``expanded_x = Expand(x, expand_shape)`` before a binary op
    ``out = BinaryOp(expanded_x, y)`` can be removed when the binary op's
    own broadcasting produces the same output shape as the explicit expand.

    The condition at each dimension ``i`` (right-aligned) is::

        max(expand_shape[i], y[i]) == max(x[i], y[i])

    which simplifies to: either ``x[i] == expand_shape[i]`` (expand is a no-op
    here) or ``y[i] == expand_shape[i]`` (y already covers the expansion).

    This check works with dynamic (symbolic) dimensions in x or y as long as
    the expand target shape is a compile-time constant.

    Args:
        expand_input: The value fed into the Expand node.
        shape: The target shape operand of the Expand node (must be a constant).
        other_input: The other operand of the binary op.

    Returns:
        A MatchResult that is successful when the Expand can be removed.
    """
    check_result = MatchResult()

    # Need at least the rank of both inputs.
    expand_input_shape = expand_input.shape
    other_shape = other_input.shape
    if expand_input_shape is None or other_shape is None:
        return check_result.fail("Input shapes are not known.")

    # The Expand target shape must be a compile-time constant.
    expand_shape_val = get_numpy_value(shape)
    if expand_shape_val is None:
        return check_result.fail("Expand target shape is not a constant.")

    expand_shape = tuple(int(v) for v in expand_shape_val.tolist())
    expand_rank = len(expand_shape)
    x_rank = expand_input_shape.rank()
    y_rank = other_shape.rank()

    # Check each dimension of expand_shape (right-aligned).
    # For the expand to be removable at position i, we need:
    #   max(e_d, y_d) == max(x_d, y_d)
    # which requires: e_d <= max(x_d, y_d).
    # Since a valid Expand can only broadcast from 1 (not shrink), if e_d > 1
    # then x_d is either 1 or e_d. The condition then reduces to:
    #   x_d == e_d  OR  y_d == e_d.
    for rev_i in range(expand_rank):
        i = expand_rank - 1 - rev_i
        e_d = expand_shape[i]  # always a known integer

        # If expand target is 1 at this dim, expand cannot shrink a dimension, so
        # x_d must also be 1. The output is max(1, y_d) = y_d in both cases.
        if e_d == 1:
            continue

        # Get x dimension (virtually 1 if x has fewer dims than expand_shape).
        x_idx = x_rank - 1 - rev_i
        x_d = expand_input_shape[x_idx] if x_idx >= 0 else 1

        # If x's dimension already equals the expand target, expand is a no-op here.
        if isinstance(x_d, int) and x_d == e_d:
            continue

        # The expand is changing this dimension (x_d is 1 or symbolic).
        # For the binary op to yield the same output, y must supply this dimension.
        # Get y dimension (virtually 1 if y has fewer dims than expand_shape).
        y_idx = y_rank - 1 - rev_i
        y_d = other_shape[y_idx] if y_idx >= 0 else 1

        if isinstance(y_d, int) and y_d == e_d:
            continue  # y covers the expansion at this dimension

        return check_result.fail(
            f"Cannot verify that removing Expand is safe at dimension {i}: "
            f"x_d={x_d!r}, expand_d={e_d}, y_d={y_d!r}."
        )

    return check_result


class _ExpandFirstInput(RewriteRuleClassBase):
    """Removes ``BinaryOp(Expand(x, shape), y)`` -> ``BinaryOp(x, y)``."""

    def __init__(self, op_type: str) -> None:
        super().__init__(f"ExpandFirst_{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x: ir.Value, shape: ir.Value, y: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(op.Expand(x, shape), y)

    def check(self, context, x: ir.Value, shape: ir.Value, y: ir.Value) -> MatchResult:
        del context  # Unused
        return _check_expand_removable(x, shape, y)

    def rewrite(self, op, x: ir.Value, shape: ir.Value, y: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, y)


class _ExpandSecondInput(RewriteRuleClassBase):
    """Removes ``BinaryOp(x, Expand(y, shape))`` -> ``BinaryOp(x, y)``."""

    def __init__(self, op_type: str) -> None:
        super().__init__(f"ExpandSecond_{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x: ir.Value, y: ir.Value, shape: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, op.Expand(y, shape))

    def check(self, context, x: ir.Value, y: ir.Value, shape: ir.Value) -> MatchResult:
        del context  # Unused
        return _check_expand_removable(y, shape, x)

    def rewrite(self, op, x: ir.Value, y: ir.Value, shape: ir.Value) -> ir.Value:
        return getattr(op, self._op_type)(x, y)


def _make_expand_before_binary_op_rules() -> list:
    """Create rewrite rules for removing Expand before each supported binary op."""
    rules = []
    for op_type in _BROADCAST_BINARY_OPS:
        rules.append(_ExpandFirstInput.rule(op_type))
        rules.append(_ExpandSecondInput.rule(op_type))
    return rules


expand_before_binary_op_rules = RewriteRuleSet(_make_expand_before_binary_op_rules())
