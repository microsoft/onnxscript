# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import onnxscript.rewriter.pattern as orp
from onnxscript import ir


class FusedMatMulDiv1(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Div`` with FusedMatMul."""

    def pattern(self, op, x, y, cst):
        return op.Div(op.MatMul(x, y), cst)

    def check(self, context, x, y, cst) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if cst.const_value is None:
            return check_result.fail("Divisor is not a constant value.")
        value = cst.const_value.numpy()
        if value.size > 1:
            return check_result.fail("Divisor is not a scalar value.")
        return check_result

    def rewrite(self, op, x, y, cst):
        value = cst.const_value.numpy()
        c = float(value[0] if value.shape == (1,) else value)
        return op.FusedMatMul(x, y, alpha=1 / c, _domain="com.microsoft")


class FusedMatMulDiv2(orp.RewriteRuleClassBase):
    """Replaces ``FusedMatMul + Div`` with FusedMatMul."""

    def pattern(self, op, x, y, cst):
        return op.Div(op.FusedMatMul(x, y, _domain="com.microsoft", _outputs=["fused"]), cst)

    def check(self, context, x, y, cst, fused: ir.Value) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if cst.const_value is None:
            return check_result.fail("Divisor is not a constant value.")
        if cst.const_value.numpy().size > 1:
            return check_result.fail("Divisor is not a scalar value.")
        return check_result

    def rewrite(self, op, x, y, cst, fused: ir.Value):
        value = cst.const_value.numpy()
        c = float(value[0] if value.shape == (1,) else value)
        node = fused.producer()
        assert node is not None, "FusedMatMul node should not be None"
        kwargs = {key: val.value for key, val in node.attributes.items()}
        kwargs["alpha"] = node.attributes["alpha"].as_float() / c
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class _TransposeMatMulBase(orp.RewriteRuleClassBase):
    _pos: ClassVar = 1

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value | None = None, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        node = transposed.producer()
        assert node is not None, "Transpose node should not be None"
        perm = node.attributes["perm"].as_ints()
        # Check that last two dimensions are swapped
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        if perm != expected_perm:
            return check_result.fail("Permutation values for Transpose are not correct.")
        if fused:
            fused_node = fused.producer()
            assert fused_node is not None, "FusedMatMul node should not be None"
            if fused_node.attributes["transBatchA"].as_int() == 1 and self._pos == 2:
                return check_result.fail(
                    "FusedMatMul with transBatchA cannot be used with Transpose(A)."
                )
            if fused_node.attributes["transBatchB"].as_int() == 1 and self._pos == 1:
                return check_result.fail(
                    "FusedMatMul with transBatchB cannot be used with Transpose(B)."
                )
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            node = fused.producer()
            assert node is not None, "FusedMatMul node should not be None"
            kwargs = {key: val.value for key, val in node.attributes.items()}
        name = "transA" if self._pos == 1 else "transB"
        kwargs[name] = 1 - kwargs.get(name, 0)
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class TransposeMatMul1(_TransposeMatMulBase):
    """Replaces ``Transpose + MatMul`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.MatMul(op.Transpose(x, _outputs=["transposed"]), y)


class TransposeFusedMatMul1(TransposeMatMul1):
    """Replaces ``Transpose + (Fused)MatMul`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.FusedMatMul(
            op.Transpose(x, _outputs=["transposed"]),
            y,
            _domain="com.microsoft",
            _outputs=["fused"],
        )


class TransposeMatMul2(_TransposeMatMulBase):
    """Replaces ``Transpose + MatMul`` with FusedMatMul."""

    _pos: ClassVar = 2

    def pattern(self, op, x, y):
        return op.MatMul(x, op.Transpose(y, _outputs=["transposed"]))


class TransposeFusedMatMul2(TransposeMatMul2):
    """Replaces ``Transpose + (Fused)MatMul`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.FusedMatMul(
            x,
            op.Transpose(y, _outputs=["transposed"]),
            _domain="com.microsoft",
            _outputs=["fused"],
        )


class _TransposeFusedMatMulBaseWithBatch(orp.RewriteRuleClassBase):
    """Base class for Transpose + FusedMatMul with batch transpose support."""

    _pos: ClassVar = 1
    _flip_transpose_batch: ClassVar = False
    _flip_transpose: ClassVar = False

    def rewrite(self, op, x, y, fused: ir.Value, **_):
        kwargs = {}
        node = fused.producer()
        assert node is not None, "FusedMatMul node should not be None"
        kwargs = {key: val.value for key, val in node.attributes.items()}
        name = "A" if self._pos == 1 else "B"
        if self._flip_transpose_batch:
            transBatchName = f"transBatch{name}"
            kwargs[transBatchName] = 1 - kwargs[transBatchName]
        if self._flip_transpose:
            transName = f"trans{name}"
            kwargs[transName] = 1 - kwargs[transName]
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")

    def pattern(self, op, x, y):
        if self._pos == 1:
            return op.FusedMatMul(
                op.Transpose(x, _outputs=["transposed"]),
                y,
                _domain="com.microsoft",
                _outputs=["fused"],
            )
        else:
            return op.FusedMatMul(
                x,
                op.Transpose(y, _outputs=["transposed"]),
                _domain="com.microsoft",
                _outputs=["fused"],
            )


class TransposeFusedMatMulWithFlippedBatch1(_TransposeFusedMatMulBaseWithBatch):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when only transBatchA can be flipped i.e.,
    when the transpose indices are [1:-1, 0, -1] for transBatchA = 0 and
    [-2, 0:-2, -1] for transBatchA = 1.
    """

    _flip_transpose_batch: ClassVar = True

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        node = transposed.producer()
        assert node is not None, "Transpose node should not be None"
        fused_node = fused.producer()
        assert fused_node is not None, "FusedMatMul node should not be None"
        perm = node.attributes["perm"].as_ints()
        # Check that last two dimensions are swapped
        list_perm = list(range(len(perm)))
        expected_perm0 = list_perm[1:-1] + [list_perm[0], list_perm[-1]]
        expected_perm1 = [list_perm[-2]] + list_perm[0:-2] + [list_perm[-1]]
        if self._pos == 1:
            property = "transBatchA"
        else:
            property = "transBatchB"
        transBatch = fused_node.attributes[property].as_int()
        if (expected_perm0 == perm and transBatch == 0) or (
            expected_perm1 == perm and transBatch == 1
        ):
            return check_result
        return check_result.fail("Permutation values for Transpose are not correct.")


class TransposeFusedMatMulWithFlippedBatch2(_TransposeFusedMatMulBaseWithBatch):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when only transBatchB can be flipped i.e.,
    when the transpose indices are [1:-1, 0, -1] for transBatchB = 0 and
    [-2, 0:-2, -1] for transBatchB = 1.
    """

    _pos: ClassVar = 2


class TransposeFusedMatMulWithFlippedBatchAndTranspose1(_TransposeFusedMatMulBaseWithBatch):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when transBatchA and transA can be flipped i.e.,
    when the transpose indices are [1:-1, -1, 0] for transBatchA = 0 and
    [-2, 0:-2, -1] for transBatchA = 1.
    """

    _flip_transpose_batch: ClassVar = True
    _flip_transpose: ClassVar = True

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        node = transposed.producer()
        assert node is not None, "Transpose node should not be None"
        fused_node = fused.producer()
        assert fused_node is not None, "FusedMatMul node should not be None"
        perm = node.attributes["perm"].as_ints()
        # Check that last two dimensions are swapped
        list_perm = list(range(len(perm)))
        expected_perm0 = list_perm[1:] + [list_perm[0]]
        expected_perm1 = [list_perm[-1]] + list_perm[0:-1]
        if self._pos == 1:
            property = "transBatchA"
        else:
            property = "transBatchB"
        transBatch = fused_node.attributes[property].as_int()
        if (expected_perm0 == perm and transBatch == 0) or (
            expected_perm1 == perm and transBatch == 1
        ):
            return check_result
        return check_result.fail("Permutation values for Transpose are not correct.")


class TransposeFusedMatMulWithFlippedBatchAndTranspose2(
    TransposeFusedMatMulWithFlippedBatchAndTranspose1
):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when transBatchB and transB can be flipped i.e.,
    when the transpose indices are [1:-1, -1, 0] for transBatchB = 0 and
    [-2, 0:-2, -1] for transBatchB = 1.
    """

    _pos: ClassVar = 2


class TransposeFusedMatMulWithBatchAndTranspose1(_TransposeFusedMatMulBaseWithBatch):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when transBatchA = 1 and transA can be flipped i.e.,
    when the transpose indices are [-1, 1:-1, 0].
    """

    _flip_transpose: ClassVar = True

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        node = transposed.producer()
        assert node is not None, "Transpose node should not be None"
        fused_node = fused.producer()
        assert fused_node is not None, "FusedMatMul node should not be None"
        perm = node.attributes["perm"].as_ints()
        # Check that last two dimensions are swapped
        list_perm = list(range(len(perm)))
        expected_perm = [list_perm[-1]] + list_perm[1:-1] + [list_perm[0]]
        if self._pos == 1:
            property = "transBatchA"
        else:
            property = "transBatchB"
        transBatch = fused_node.attributes[property].as_int()
        if expected_perm == perm and transBatch == 1:
            return check_result
        return check_result.fail("Permutation values for Transpose are not correct.")


class TransposeFusedMatMulWithBatchAndTranspose2(TransposeFusedMatMulWithBatchAndTranspose1):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul.
    This rule is for when transBatchB = 1 and transB can be flipped i.e.,
    when the transpose indices are [-1, 1:-1, 0].
    """

    _pos: ClassVar = 2


class MatMulTranspose(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Transpose`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.Transpose(op.MatMul(x, y), _outputs=["transposed"])

    def check(self, context, x, y, transposed: ir.Value, **_) -> orp.MatchResult:
        check_result = orp.MatchResult()
        transpose = transposed.producer()
        assert transpose is not None, "Transpose node should not be None"
        perm = transpose.attributes["perm"].as_ints()
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        if perm != expected_perm:
            return check_result.fail("Permutation values for Transpose are not correct.")
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            node = fused.producer()
            assert node is not None, "FusedMatMul node should not be None"
            kwargs = {key: val.value for key, val in node.attributes.items()}
        for name in ["transA", "transB"]:
            kwargs[name] = 1 - kwargs.get(name, 0)
        return op.FusedMatMul(y, x, **kwargs, _domain="com.microsoft")


class FusedMatMulTranspose(MatMulTranspose):
    """Replaces ``FusedMatMul + Transpose`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.Transpose(
            op.FusedMatMul(x, y, _domain="com.microsoft", _outputs=["fused"]),
            _outputs=["transposed"],
        )


def fused_matmul_rule_sets() -> orp.RewriteRuleSet:
    """Returns a set of rules introducing onnxruntime contrib obs.
    This requires onnxruntime to run the model after
    it is rewritten.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            FusedMatMulDiv1.rule(),
            FusedMatMulDiv2.rule(),
            FusedMatMulTranspose.rule(),
            MatMulTranspose.rule(),
            TransposeMatMul1.rule(),
            TransposeFusedMatMul1.rule(),
            TransposeMatMul2.rule(),
            TransposeFusedMatMul2.rule(),
            TransposeFusedMatMulWithFlippedBatch1.rule(),
            TransposeFusedMatMulWithFlippedBatch2.rule(),
            TransposeFusedMatMulWithFlippedBatchAndTranspose1.rule(),
            TransposeFusedMatMulWithFlippedBatchAndTranspose2.rule(),
            TransposeFusedMatMulWithBatchAndTranspose1.rule(),
            TransposeFusedMatMulWithBatchAndTranspose2.rule(),
        ]
    )
