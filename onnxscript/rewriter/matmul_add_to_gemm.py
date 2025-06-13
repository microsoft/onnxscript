# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Does the following transformation:
- Add(MatMul(X, W), B) -> Gemm
- Add(MatMul(Transpose(X), W), B) -> Gemm
- Add(MatMul(X, Transpose(W)), B) -> Gemm
- Add(MatMul(Transpose(X), Transpose(W)), B) -> Gemm
"""

import abc
from typing import ClassVar

from onnxscript.rewriter import pattern as orp


class _MatMulAddToGemmBase(orp.RewriteRuleClassBase, abc.ABC):
    trans_a: ClassVar = False
    trans_b: ClassVar = False

    def rewrite(self, op, input_a, input_b, input_c):
        attributes = {}
        if self.trans_a:
            attributes["transA"] = 1
        if self.trans_b:
            attributes["transB"] = 1
        return op.Gemm(input_a, input_b, input_c, **attributes)

    def check(self, context, input_a, input_b, **_):
        del context  # Not used
        check_result = orp.MatchResult()
        # Rank of input_a and input_b must be 2
        if len(input_a.shape) != 2 or len(input_b.shape) != 2:
            return check_result.fail("Rank of input_a and input_b must be 2")
        return check_result


class MatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(a, b), c)`` with ``Gemm(a, b, c)``."""

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(input_a, input_b)
        return op.Add(matmul, input_c)


class TransAMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(Transpose(a), b), c)`` with ``Gemm(a, b, c)``."""

    trans_a: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(op.Transpose(input_a, perm=[1, 0]), input_b)
        return op.Add(matmul, input_c)


class TransBMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(a, Transpose(b)), c)`` with ``Gemm(a, b, c)``."""

    trans_b: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(input_a, op.Transpose(input_b, perm=[1, 0]))
        return op.Add(matmul, input_c)


class TransABMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(Transpose(a), Transpose(b)), c)`` with ``Gemm(a, b, c)``."""

    trans_a: ClassVar = True
    trans_b: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(
            op.Transpose(input_a, perm=[1, 0]),
            op.Transpose(input_b, perm=[1, 0]),
        )
        return op.Add(matmul, input_c)


matmul_add_to_gemm_rule = MatMulAddToGemm().rule()
transpose_a_matmul_add_to_gemm_rule = TransAMatMulAddToGemm().rule()
transpose_b_matmul_add_to_gemm_rule = TransBMatMulAddToGemm().rule()
transpose_ab_matmul_add_to_gemm_rule = TransABMatMulAddToGemm().rule()


def gemm_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rewrite rules that fuse MatMul + Add patterns into a single Gemm node,
    handling cases where one or both MatMul inputs are transposed.

    Returns:
        RewriteRuleSet
    """

    # Order is important
    return orp.RewriteRuleSet(
        [
            transpose_ab_matmul_add_to_gemm_rule,
            transpose_a_matmul_add_to_gemm_rule,
            transpose_b_matmul_add_to_gemm_rule,
            matmul_add_to_gemm_rule,
        ]
    )
