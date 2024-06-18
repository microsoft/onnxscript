# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.rewriter.onnxruntime as oort
import onnxscript.rewriter.pattern as orp

op = orp.onnxop


class FusedMatMulDiv1(orp.RewriteRuleAsClass):
    """Replaces ``MatMul + Div`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y, cst):
        return op.Div(op.MatMul(x, y), cst)

    @classmethod
    def check(cls, context, x, y, cst) -> bool:
        if cst.const_value is None:
            return False
        value = cst.const_value.numpy()
        if value.size > 1:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, y, cst):
        value = cst.const_value.numpy()
        c = float(value[0] if value.shape == (1,) else value)
        return op.FusedMatMul(x, y, alpha=1 / c, domain="com.microsoft")


class FusedMatMulDiv2(orp.RewriteRuleAsClass):
    """Replaces ``FusedMatMul + Div`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y, cst):
        return op.Div(op.FusedMatMul(x, y, domain="com.microsoft"), cst)

    @classmethod
    def check(cls, context, x, y, cst) -> bool:
        if cst.const_value is None:
            return False
        if cst.const_value.numpy().size > 1:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, y, cst):
        value = cst.const_value.numpy()
        c = float(value[0] if value.shape == (1,) else value)
        nodes = list(x.uses())
        assert (
            len(nodes) == 1
        ), f"The pattern should not match if x is used {len(nodes)} times."
        node = nodes[0][0]

        kwargs = {}
        alpha = node.attributes.get("alpha", None)
        kwargs["alpha"] = alpha = alpha.value / c if alpha else 1.0 / c
        for name in ["transA", "transB", "transBatchA", "transBatchB"]:
            att = node.attributes.get(name)
            if att:
                kwargs[name] = att.value
        return op.FusedMatMul(x, y, **kwargs, domain="com.microsoft")


def fused_matmul_rule_sets() -> orp.RewriteRuleSet:
    """Returns a set of rules introducting onnxruntime contrib obs.
    This requires onnxruntime to run the model after
    it is rewritten.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            *oort.ORT_PATTERN_REWRITE_RULES,
            orp.make_rewrite_rule_from_class(FusedMatMulDiv1, True),
            orp.make_rewrite_rule_from_class(FusedMatMulDiv2, True),
        ]
    )
