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

import numpy as np

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

    The Expand is removable if the binary op's broadcasting produces the same
    output shape when using the original (pre-expand) tensor directly.

    Args:
        expand_input: The value fed into the Expand node.
        shape: The target shape operand of the Expand node (must be a constant).
        other_input: The other operand of the binary op.

    Returns:
        A MatchResult that is successful when the Expand can be removed.
    """
    check_result = MatchResult()

    # Need static shape info for both inputs.
    expand_input_shape = expand_input.shape
    other_shape = other_input.shape
    if expand_input_shape is None or other_shape is None:
        return check_result.fail("Input shapes are not statically known.")

    # Require fully static (integer-only) shapes to avoid symbolic dim issues.
    if not expand_input_shape.is_static() or not other_shape.is_static():
        return check_result.fail("Input shapes are not fully static.")

    # The Expand target shape must be a compile-time constant.
    expand_shape_val = get_numpy_value(shape)
    if expand_shape_val is None:
        return check_result.fail("Expand target shape is not a constant.")

    expand_shape = tuple(int(v) for v in expand_shape_val.tolist())
    x_shape = tuple(int(d) for d in expand_input_shape)
    y_shape = tuple(int(d) for d in other_shape)

    # Verify that removing the Expand does not change the binary op's output shape.
    try:
        result_with_expand = np.broadcast_shapes(expand_shape, y_shape)
        result_without_expand = np.broadcast_shapes(x_shape, y_shape)
    except ValueError:
        return check_result.fail("Shapes are not broadcastable.")

    if result_with_expand != result_without_expand:
        return check_result.fail(
            f"Removing Expand would change output shape from "
            f"{result_with_expand} to {result_without_expand}."
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
