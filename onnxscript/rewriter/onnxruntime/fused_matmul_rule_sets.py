# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import onnxscript.rewriter.pattern as orp


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
        return op.FusedMatMul(x, y, alpha=1 / c, _domain="com.microsoft")


class FusedMatMulDiv2(orp.RewriteRuleAsClass):
    """Replaces ``FusedMatMul + Div`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y, cst):
        return op.Div(op.FusedMatMul(x, y, _domain="com.microsoft"), cst)

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
        node = list(x.uses())[0][0]  # noqa: RUF015

        kwargs = {}
        alpha = node.attributes.get("alpha", None)
        kwargs["alpha"] = alpha.value / c if alpha else 1.0 / c
        for name in ["transA", "transB", "transBatchA", "transBatchB"]:
            att = node.attributes.get(name)
            if att:
                kwargs[name] = att.value
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class _TransposeMatMulBase(orp.RewriteRuleAsClass):
    _pos: ClassVar = 1

    @classmethod
    def check(cls, context, x, y) -> bool:
        perm = list((x if cls._pos == 1 else y).uses())[0][0].attributes["perm"].value  # noqa: RUF015
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        return perm == expected_perm

    @classmethod
    def rewrite(cls, op, x, y):
        node = list((x if cls._pos == 2 else y).uses())[0][0]  # noqa: RUF015
        kwargs = {}
        for name in ["alpha", "transA", "transB", "transBatchA", "transBatchB"]:
            att = node.attributes.get(name)
            if att:
                kwargs[name] = att.value
        name = "transA" if cls._pos == 1 else "transB"
        kwargs[name] = 1 - kwargs.get(name, 0)
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class TransposeMatMul1(_TransposeMatMulBase):
    """Replaces ``Transpose + (Fused)MatMul`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y):
        return op.MatMul(op.Transpose(x), y)


class TransposeFusedMatMul1(TransposeMatMul1):
    """Replaces ``Transpose + (Fused)MatMul`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y):
        return op.FusedMatMul(op.Transpose(x), y, _domain="com.microsoft")


class TransposeMatMul2(_TransposeMatMulBase):
    """Replaces ``Transpose + (Fused)MatMul`` by FusedMatMul."""

    _pos: ClassVar = 2

    @classmethod
    def pattern(cls, op, x, y):
        return op.MatMul(x, op.Transpose(y))


class TransposeFusedMatMul2(TransposeMatMul2):
    """Replaces ``Transpose + (Fused)MatMul`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y):
        return op.FusedMatMul(x, op.Transpose(y), _domain="com.microsoft")


class MatMulTranspose(orp.RewriteRuleAsClass):
    """Replaces ``MatMul + Transpose`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y):
        return op.Transpose(op.MatMul(x, y))

    @classmethod
    def check(cls, context, x, y) -> bool:
        matmul = list(x.uses())[0][0]  # noqa: RUF015
        transpose = list(matmul.outputs[0].uses())[0][0]  # noqa: RUF015
        perm = transpose.attributes["perm"].value
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        return perm == expected_perm

    @classmethod
    def rewrite(cls, op, x, y):
        node = list(x.uses())[0][0]  # noqa: RUF015
        kwargs = {}
        for name in ["alpha", "transA", "transB", "transBatchA", "transBatchB"]:
            att = node.attributes.get(name)
            if att:
                kwargs[name] = att.value
        for name in ["transA", "transB"]:
            kwargs[name] = 1 - kwargs.get(name, 0)
        return op.FusedMatMul(y, x, **kwargs, _domain="com.microsoft")


class FusedMatMulTranspose(MatMulTranspose):
    """Replaces ``MatMul + Transpose`` by FusedMatMul."""

    @classmethod
    def pattern(cls, op, x, y):
        return op.Transpose(op.FusedMatMul(x, y, _domain="com.microsoft"))


def fused_matmul_rule_sets() -> orp.RewriteRuleSet:
    """Returns a set of rules introducting onnxruntime contrib obs.
    This requires onnxruntime to run the model after
    it is rewritten.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            orp.make_rewrite_rule_from_class(FusedMatMulDiv1, True),
            orp.make_rewrite_rule_from_class(FusedMatMulDiv2, True),
            orp.make_rewrite_rule_from_class(FusedMatMulTranspose, True),
            orp.make_rewrite_rule_from_class(MatMulTranspose, True),
            orp.make_rewrite_rule_from_class(TransposeMatMul1, True),
            orp.make_rewrite_rule_from_class(TransposeFusedMatMul1, True),
            orp.make_rewrite_rule_from_class(TransposeMatMul2, True),
            orp.make_rewrite_rule_from_class(TransposeFusedMatMul2, True),
        ]
    )
