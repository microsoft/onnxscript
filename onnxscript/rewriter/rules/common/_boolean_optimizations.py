# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Exact Boolean and selection simplifications."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import get_numpy_value, same_shape
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet
from onnxscript.rewriter.rules.common._remove_expand_before_binary_op import (
    _compute_broadcast_shape,
)


class NotNot(RewriteRuleClassBase):
    def pattern(self, op, x):
        return op.Not(op.Not(x))

    def rewrite(self, op, x):
        return op.Identity(x)


class NotWhere(RewriteRuleClassBase):
    def pattern(self, op, condition, x, y):
        return op.Where(op.Not(condition), x, y)

    def rewrite(self, op, condition, x, y):
        return op.Where(condition, y, x)


class BooleanIdentity(RewriteRuleClassBase):
    def __init__(self, op_type: str, neutral: bool):
        self.op_type = op_type
        self.neutral = neutral
        super().__init__(f"{op_type}Identity")

    def pattern(self, op, x, constant):
        return getattr(op, self.op_type)(x, constant)

    def check(self, context, x: ir.Value, constant: ir.Value) -> MatchResult:
        result = MatchResult()
        value = get_numpy_value(constant)
        if value is None or value.dtype != np.bool_ or not np.all(value == self.neutral):
            return result.fail("The Boolean operand is not a constant neutral value.")
        if value.ndim == 0:
            return result
        if x.shape is None:
            return result.fail("Input rank and shape are needed for a nonscalar constant.")
        broadcast = _compute_broadcast_shape(x.shape, ir.Shape(value.shape))
        if broadcast is None or not same_shape(x.shape, ir.Shape(broadcast)):
            return result.fail("The constant expands the input shape.")
        return result

    def rewrite(self, op, x, constant):
        return op.Identity(x)


class WhereAdd(RewriteRuleClassBase):
    """Factor an Add without reassociating or exchanging its operands."""

    def __init__(self, common_first: bool):
        self.common_first = common_first
        super().__init__(f"WhereAddCommon{'First' if common_first else 'Second'}")

    def pattern(self, op, condition, common, x, y):
        if self.common_first:
            return op.Where(condition, op.Add(common, x), op.Add(common, y))
        return op.Where(condition, op.Add(x, common), op.Add(y, common))

    def rewrite(self, op, condition, common, x, y):
        selected = op.Where(condition, x, y)
        if self.common_first:
            return op.Add(common, selected)
        return op.Add(selected, common)


def _axes(value: ir.Value, rank: int) -> list[int] | None:
    axes = get_numpy_value(value)
    if axes is None or axes.dtype != np.int64 or axes.ndim != 1:
        return None
    result = [int(a) + rank if a < 0 else int(a) for a in axes]
    if any(a < 0 or a >= rank for a in result) or len(set(result)) != len(result):
        return None
    return sorted(result)


class UnsqueezeEqual(RewriteRuleClassBase):
    def pattern(self, op, x, y, axes_x, axes_y):
        return op.Equal(op.Unsqueeze(x, axes_x), op.Unsqueeze(y, axes_y))

    def check(self, context, x, y, axes_x, axes_y) -> MatchResult:
        result = MatchResult()
        if x.shape is None or y.shape is None or len(x.shape) != len(y.shape):
            return result.fail("The pre-Unsqueeze ranks must be known and equal.")
        raw_axes = get_numpy_value(axes_x)
        if raw_axes is None or raw_axes.ndim != 1:
            return result.fail("Unsqueeze axes must be constant vectors.")
        rank = len(x.shape) + raw_axes.size
        normalized = _axes(axes_x, rank)
        if normalized is None or normalized != _axes(axes_y, rank):
            return result.fail("Unsqueeze axes must be valid and equivalent.")
        return result

    def rewrite(self, op, x, y, axes_x, axes_y):
        return op.Unsqueeze(op.Equal(x, y), axes_x)


class ReuseUnsqueezedComparison(RewriteRuleClassBase):
    def pattern(self, op, x, constant, axes):
        return op.Unsqueeze(op.Equal(x, constant), axes)

    def check(self, context, x, constant, axes) -> MatchResult:
        result = MatchResult()
        value = get_numpy_value(constant)
        if value is None or value.size != 1:
            return result.fail("The compared operand must be a constant singleton.")
        if x.shape is None or value.ndim > len(x.shape):
            return result.fail("The comparison must preserve the input rank.")
        raw_axes = get_numpy_value(axes)
        if raw_axes is None or raw_axes.ndim != 1:
            return result.fail("Unsqueeze axes must be a constant vector.")
        rank = len(x.shape) + raw_axes.size
        normalized = _axes(axes, rank)
        if normalized is None:
            return result.fail("Unsqueeze axes are invalid.")
        available = set()
        for node in context.root.graph:
            if node is context.root:
                break
            available.add(node)
        for use in x.uses():
            node = use.node
            if (
                node in available
                and node.domain == ""
                and node.op_type == "Unsqueeze"
                and len(node.inputs) == 2
                and node.inputs[1] is not None
                and _axes(node.inputs[1], rank) == normalized
            ):
                self._unsqueezed = node.outputs[0]
                return result
        return result.fail("There is no equivalent existing Unsqueeze of the input.")

    def rewrite(self, op, x, constant, axes):
        value = get_numpy_value(constant)
        scalar = op.Constant(value=ir.tensor(value.reshape(())))
        return op.Equal(self._unsqueezed, scalar)


not_not_rule = NotNot.rule(remove_nodes=False)
not_where_rule = NotWhere.rule(remove_nodes=False)
and_identity_rule = BooleanIdentity.rule("And", True)
or_identity_rule = BooleanIdentity.rule("Or", False)
where_add_first_rule = WhereAdd.rule(True)
where_add_second_rule = WhereAdd.rule(False)
unsqueeze_equal_rule = UnsqueezeEqual.rule()
reuse_unsqueezed_comparison_rule = ReuseUnsqueezedComparison.rule()

rules = RewriteRuleSet(
    [
        not_not_rule,
        not_where_rule,
        *and_identity_rule.commute(),
        *or_identity_rule.commute(),
        where_add_first_rule,
        where_add_second_rule,
        unsqueeze_equal_rule,
        reuse_unsqueezed_comparison_rule,
    ]
)
