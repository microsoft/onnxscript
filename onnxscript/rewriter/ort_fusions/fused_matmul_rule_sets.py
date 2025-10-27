# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import onnxscript.rewriter.pattern as orp
from onnxscript import ir
from onnxscript.rewriter import _ir_utils


def _get_node(value: ir.Value, name: str) -> ir.Node:
    """Get the node from the output value."""
    node = value.producer()
    assert node is not None, f"{name} node should not be None"
    return node


def _get_kwargs(node: ir.Node) -> dict[str, float | int]:
    """Get the kwargs from the node."""
    kwargs = {key: val.value for key, val in node.attributes.items()}
    return kwargs


class FusedMatMulDiv1(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Div`` with MatMul."""

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

    def check(self, context, x, y, cst, **_) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if cst.const_value is None:
            return check_result.fail("Divisor is not a constant value.")
        if cst.const_value.numpy().size > 1:
            return check_result.fail("Divisor is not a scalar value.")
        return check_result

    def rewrite(self, op, x, y, cst, fused: ir.Value):
        value = cst.const_value.numpy()
        c = float(value[0] if value.shape == (1,) else value)
        fused_node = _get_node(fused, "FusedMatMul")
        kwargs = _get_kwargs(fused_node)
        kwargs["alpha"] = kwargs.get("alpha", 1.0) / c
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class _TransposeMatMulBase(orp.RewriteRuleClassBase):
    _pos: ClassVar = 1

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value | None = None, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        transposed_node = _get_node(transposed, "Transpose")
        perm = transposed_node.attributes.get_ints("perm")
        if perm:
            # Check that last two dimensions are swapped
            expected_perm = list(range(len(perm)))
            expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
            if list(perm) != expected_perm:
                return check_result.fail("Permutation values for Transpose are not correct.")
        elif (self._pos == 1 and not _ir_utils.has_rank(x, 2)) or (
            self._pos == 2 and not _ir_utils.has_rank(y, 2)
        ):
            # If perm is not defined, the default transpose behavior is to swap
            #   all dimensions, which is correct for MatMul with rank = 2.
            return check_result.fail(
                "If perm is not defined, rank must be 2 for TransposeMatMul rule."
            )
        if fused:
            fused_node = _get_node(fused, "FusedMatMul")
            trans_batch_property = "transBatchA" if self._pos == 1 else "transBatchB"
            if fused_node.attributes.get_int(trans_batch_property, 0):
                return check_result.fail(
                    "FusedMatMul with transposed batch cannot be used with op.Transpose in this rule."
                )
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            fused_node = _get_node(fused, "FusedMatMul")
            kwargs = _get_kwargs(fused_node)
        trans_name = "transA" if self._pos == 1 else "transB"
        kwargs[trans_name] = 1 - kwargs.get(trans_name, 0)
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class TransposeMatMul1(_TransposeMatMulBase):
    """Replaces ``Transpose + MatMul`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.MatMul(op.Transpose(x, _outputs=["transposed"]), y)


class TransposeFusedMatMul1(TransposeMatMul1):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul."""

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
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.FusedMatMul(
            x,
            op.Transpose(y, _outputs=["transposed"]),
            _domain="com.microsoft",
            _outputs=["fused"],
        )


class _TransposeFusedMatMulBaseWithBatch(orp.RewriteRuleClassBase):
    """Replaces ``Transpose + FusedMatMul`` with FusedMatMul, either
       when transBatchA or transBatchB in FusedMatMul is 1, or
       can be inverted based on the permutation dims of the Transpose, in
       contrast to the original FusedMatMul rule which assumes that
       transBatchA and transBatchB are always 0 before and after rewriting.

    transBatchA = 1, transA = 0 applies a batch transpose by moving the first dimension to the second-to-last position
        i.e., equivalent to a Transpose with "perm" [1, 2, ..., N-2, 0, N-1].
    transBatchA = 0, transA = 1 flips the last two dimensions
        i.e., equivalent to a Transpose with "perm" [0, 1, ... N-3, N-1, N-2].
    transBatchA = 1, transA = 1 applies a batch transpose, then flips the last two dimensions
        i.e., equivalent to a Transpose with "perm" [1, 2, ..., N-1, 0].

    The flipping logic is based on the following cases:
       Case 1: transBatchA is 0, Transpose "perm" is [1, 2, ..., N-1, 0]
            or transBatchA is 1, Transpose "perm" is [N-1, 0, 1, ..., N-2]
        - Then transBatchA and transA can be flipped in FusedMatMul when rewriting.
       Case 2: transBatchA is 0, Transpose "perm" is [1, 2, ..., N-2, 0, N-1]
            or transBatchA is 1, Transpose "perm" is [N-2, 0, 1, ..., N-3, N-1]
        - Then transBatchA can be flipped in FusedMatMul when rewriting.
       Case 3: transBatchA is 1, Transpose "perm" is [N-1, 1, ..., N-2, 0]
        - Then transA can be flipped in FusedMatMul when rewriting.
    The same logic applies for transBatchB and transB, when _pos is set to 2.
    The _flip_transpose_batch and _flip_transpose flags are used to control
    which case is applied by the rules of inheriting classes that change these class vars.
    """

    _pos: ClassVar = 1
    _flip_transpose_batch: ClassVar = False
    _flip_transpose: ClassVar = False

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        fused_node = _get_node(fused, "FusedMatMul")
        trans_batch_property = "transBatchA" if self._pos == 1 else "transBatchB"
        trans_batch = fused_node.attributes.get_int(trans_batch_property, 0)
        transposed_node = _get_node(transposed, "Transpose")
        perm = list(transposed_node.attributes["perm"].as_ints())
        if not perm:
            return check_result.fail("Permutation values for Transpose are not correct.")

        list_perm = list(range(len(perm)))
        if self._flip_transpose_batch and self._flip_transpose:
            #  Case 1: transBatchA/B is 0, Transpose "perm" is [1, 2, ..., N-1, 0]
            #       or transBatchA/B is 1, Transpose "perm" is [N-1, 0, 1, ..., N-2]
            #   - Then transBatchA/B and transA/B can be flipped in FusedMatMul when rewriting.
            if trans_batch == 0:
                expected_perm = [*list_perm[1:], list_perm[0]]
            else:
                expected_perm = [list_perm[-1], *list_perm[0:-1]]
            if expected_perm == perm:
                return check_result
        elif self._flip_transpose_batch:
            #  Case 2: transBatchA/B is 0, Transpose "perm" is [1, 2, ..., N-2, 0, N-1]
            #       or transBatchA/B is 1, Transpose "perm" is [N-2, 0, 1, ..., N-3, N-1]
            #   - Then transBatchA/B can be flipped in FusedMatMul when rewriting.
            if trans_batch == 0:
                expected_perm = [*list_perm[1:-1], list_perm[0], list_perm[-1]]
            else:
                expected_perm = [list_perm[-2], *list_perm[0:-2], list_perm[-1]]
            if expected_perm == perm:
                return check_result
        elif self._flip_transpose:
            #  Case 3: transBatchA is 1, Transpose "perm" is [N-1, 1, ..., N-2, 0]
            #   - Then transA can be flipped in FusedMatMul when rewriting.
            expected_perm = [list_perm[-1], *list_perm[1:-1], list_perm[0]]
            if expected_perm == perm and trans_batch == 1:
                return check_result

        return check_result.fail("Permutation values for Transpose are not correct.")

    def rewrite(self, op, x, y, fused: ir.Value, **_):
        kwargs = {}
        fused_node = _get_node(fused, "FusedMatMul")
        kwargs = _get_kwargs(fused_node)
        name = "A" if self._pos == 1 else "B"
        if self._flip_transpose_batch:
            trans_batch_property = f"transBatch{name}"
            kwargs[trans_batch_property] = 1 - kwargs.get(trans_batch_property, 0)
        if self._flip_transpose:
            trans_property = f"trans{name}"
            kwargs[trans_property] = 1 - kwargs.get(trans_property, 0)
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


class TransposeFusedMatMulWithFlippedBatchAndTranspose1(_TransposeFusedMatMulBaseWithBatch):
    _flip_transpose = True
    _flip_transpose_batch = True


class TransposeFusedMatMulWithFlippedBatchAndTranspose2(_TransposeFusedMatMulBaseWithBatch):
    _pos = 2
    _flip_transpose = True
    _flip_transpose_batch = True


class TransposeFusedMatMulWithFlippedBatch1(_TransposeFusedMatMulBaseWithBatch):
    _flip_transpose_batch = True


class TransposeFusedMatMulWithFlippedBatch2(_TransposeFusedMatMulBaseWithBatch):
    _pos = 2
    _flip_transpose_batch = True


class TransposeFusedMatMulWithBatchAndTranspose1(_TransposeFusedMatMulBaseWithBatch):
    _flip_transpose = True


class TransposeFusedMatMulWithBatchAndTranspose2(_TransposeFusedMatMulBaseWithBatch):
    _pos = 2
    _flip_transpose = True


class MatMulTranspose(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Transpose`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.Transpose(op.MatMul(x, y), _outputs=["transposed"])

    def check(self, context, x, y, transposed: ir.Value, **_) -> orp.MatchResult:
        check_result = orp.MatchResult()
        transpose_node = _get_node(transposed, "Transpose")
        perm = transpose_node.attributes.get_ints("perm")
        # transA/transB only work on the last two dimensions of the input,
        # so we can only apply this rule if the inputs are rank 2.
        if _ir_utils.has_rank(x, 2) and _ir_utils.has_rank(y, 2):
            if perm:
                # Check that the two dimensions are swapped
                if tuple(perm) != (1, 0):
                    return check_result.fail(
                        "Permutation values for Transpose are not correct."
                    )
        # If perm is not defined, the default transpose behavior is to swap
        #   all dimensions, which is correct for MatMul with rank = 2.
        else:
            return check_result.fail("Rank must be 2 for MatMulTranspose rule.")
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            fused_node = _get_node(fused, "FusedMatMul")
            kwargs = _get_kwargs(fused_node)
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
    """Returns a set of rules introducing onnxruntime contrib ops.
    This requires onnxruntime to run the model after it is rewritten.

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
