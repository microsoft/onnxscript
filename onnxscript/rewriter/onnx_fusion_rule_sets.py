# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript import ir
import onnx.numpy_helper

import onnxscript.rewriter.pattern as orp


class SoftmaxCrossEntropyLoss(orp.RewriteRuleAsClass):
    """Replaces many nodes into ``SoftmaxCrossEntropyLoss`` if possible."""

    @classmethod
    def pattern(cls, op, X, indices):
        neq1 = op.Not(op.Equal(indices, -100))
        wh1 = op.Where(neq1, indices, 0)
        uns = op.Unsqueeze(wh1, 1)
        ge = op.GatherElements(op.LogSoftmax(X, axis=1), uns, axis=1)
        wh2 = op.Where(neq1, op.Neg(op.Squeeze(ge, 1)), 0)
        denominator = op.Cast(
            op.ReduceSum(
                op.Cast(neq1, to=onnx.TensorProto.FLOAT), keepdims=0, noop_with_empty_axes=0
            ),
            to=onnx.TensorProto.FLOAT16,
        )
        numerator = op.Cast(
            op.ReduceSum(
                op.Cast(wh2, to=onnx.TensorProto.FLOAT), keepdims=0, noop_with_empty_axes=0
            ),
            to=onnx.TensorProto.FLOAT16,
        )
        return op.Div(numerator, denominator)

    @classmethod
    def rewrite(cls, op, X, indices):
        return op.SoftmaxCrossEntropyLoss(X, indices, ignore_index=-100, reduction="mean")

    @classmethod
    def check(cls, context, X, indices) -> bool:
        if X.dtype != onnx.TensorProto.FLOAT16:
            return False
        if indices.dtype != onnx.TensorProto.INT64:
            return False
        return True


class SoftmaxCrossEntropyLossV2(orp.RewriteRuleAsClass):
    """Replaces many nodes into ``SoftmaxCrossEntropyLoss`` if possible."""

    @classmethod
    def pattern(cls, op, X, indices):
        # Another slightly different version.
        neq1 = op.Not(op.Equal(indices, -100))
        neq2 = op.Not(op.Equal(indices, -100))
        neq3 = op.Not(op.Equal(indices, -100))
        wh1 = op.Where(neq1, indices, 0)
        uns = op.Unsqueeze(wh1, 1)
        ge = op.GatherElements(
            op.LogSoftmax(X, axis=1), op.Cast(uns, to=onnx.TensorProto.INT64), axis=1
        )
        wh2 = op.Where(neq2, op.Neg(op.Squeeze(ge, 1)), 0)
        denominator = op.Cast(
            op.ReduceSum(
                op.Cast(neq3, to=ir.DataType.INT64),
                keepdims=0,
            ),
            to=onnx.TensorProto.FLOAT16,
        )
        numerator = op.Cast(
            op.ReduceSum(
                op.Cast(wh2, to=onnx.TensorProto.FLOAT),
                keepdims=0,
            ),
            to=onnx.TensorProto.FLOAT16,
        )
        return op.Div(numerator, denominator)

    @classmethod
    def rewrite(cls, op, X, indices):
        return op.SoftmaxCrossEntropyLoss(X, indices, ignore_index=-100, reduction="mean")

    @classmethod
    def check(cls, context, X, indices) -> bool:
        if X.dtype != ir.DataType.FLOAT16:
            return False
        if indices.dtype != onnx.TensorProto.INT64:
            return False
        return True


def onnx_fusion_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before any other one as they usually remove unnecessary computation
    such as the multiplication by 1 or two consecutive transpose.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            # no_op.mul_by_1_rule,
            # no_op.add_0_rule,
            # no_op.add_0_rule,
            # no_op.div_by_1_rule,
            orp.make_rewrite_rule_from_class(SoftmaxCrossEntropyLoss),
            orp.make_rewrite_rule_from_class(SoftmaxCrossEntropyLossV2),
        ]
    )
