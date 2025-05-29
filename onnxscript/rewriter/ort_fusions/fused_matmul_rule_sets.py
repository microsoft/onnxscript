# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import onnxscript.rewriter.pattern as orp
from onnxscript import ir


def get_node(value: ir.Value, name: str) -> ir.Node:
    """Get the node from the output value."""
    node = value.producer()
    assert node is not None, f"{name} node should not be None"
    return node


def get_kwargs(node: ir.Node) -> dict[str, float | int]:
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
        fused_node = get_node(fused, "FusedMatMul")
        kwargs = get_kwargs(fused_node)
        kwargs["alpha"] = fused_node.attributes["alpha"].as_float() / c  # type: ignore[assignment]
        return op.FusedMatMul(x, y, **kwargs, _domain="com.microsoft")


class _TransposeMatMulBase(orp.RewriteRuleClassBase):
    _pos: ClassVar = 1

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value | None = None, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        transposed_node = get_node(transposed, "Transpose")
        perm = transposed_node.attributes["perm"].as_ints()
        # Check that last two dimensions are swapped
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        if perm != expected_perm:
            return check_result.fail("Permutation values for Transpose are not correct.")
        if fused:
            fused_node = get_node(fused, "FusedMatMul")
            if fused_node.attributes.get("transBatchA", 0).value == 1 and self._pos == 1:  # type: ignore[union-attr]
                return check_result.fail(
                    "FusedMatMul with transBatchA cannot be used with Transpose(A)."
                )
            if fused_node.attributes.get("transBatchB", 0).value == 1 and self._pos == 2:  # type: ignore[union-attr]
                return check_result.fail(
                    "FusedMatMul with transBatchB cannot be used with Transpose(B)."
                )
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            fused_node = get_node(fused, "FusedMatMul")
            kwargs = get_kwargs(fused_node)
        name = "transA" if self._pos == 1 else "transB"
        kwargs[name] = 1 - kwargs.get(name, 0)
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
    """Base class for Transpose + FusedMatMul with batch transpose support."""

    _pos: ClassVar = 1
    _flip_transpose_batch: ClassVar = False
    _flip_transpose: ClassVar = False

    def check(
        self, context, x, y, transposed: ir.Value, fused: ir.Value, **_
    ) -> orp.MatchResult:
        check_result = orp.MatchResult()
        fused_node = get_node(fused, "FusedMatMul")
        transBatchProperty = "transBatchA" if self._pos == 1 else "transBatchB"
        transBatch = fused_node.attributes.get(transBatchProperty, 0).as_int()  # type: ignore[union-attr]
        transposed_node = get_node(transposed, "Transpose")
        perm = transposed_node.attributes["perm"].as_ints()
        list_perm = list(range(len(perm)))

        if self._flip_transpose_batch and self._flip_transpose:
            # Check when transposeBatch is 0 or 1 and transpose can be flipped i.e., when the first index is moved to the end
            # for transposeBatch = 0 or last index is moved to the front for transposeBatch = 1
            expected_perm0 = [*list_perm[1:], list_perm[0]]
            expected_perm1 = [list_perm[-1], *list_perm[0:-1]]
            if (expected_perm0 == perm and transBatch == 0) or (
                expected_perm1 == perm and transBatch == 1
            ):
                return check_result
        elif self._flip_transpose_batch:
            # Check when transposeBatch can be flipped i.e., when the first index is moved to the second-to-last position
            # for transposeBatch = 0 or second-to-last index is moved to the front for transposeBatch = 1
            expected_perm0 = [*list_perm[1:-1], list_perm[0], list_perm[-1]]
            expected_perm1 = [list_perm[-2], *list_perm[0:-2], list_perm[-1]]
            if (expected_perm0 == perm and transBatch == 0) or (
                expected_perm1 == perm and transBatch == 1
            ):
                return check_result
        elif self._flip_transpose:
            # Check when transposeBatch = 1 and transpose can be flipped i.e., when the first and last indices are flipped.
            expected_perm = [list_perm[-1], *list_perm[1:-1], list_perm[0]]
            if expected_perm == perm and transBatch == 1:
                return check_result

        return check_result.fail("Permutation values for Transpose are not correct.")

    def rewrite(self, op, x, y, fused: ir.Value, **_):
        kwargs = {}
        fused_node = get_node(fused, "FusedMatMul")
        kwargs = get_kwargs(fused_node)
        name = "A" if self._pos == 1 else "B"
        if self._flip_transpose_batch:
            transBatchName = f"transBatch{name}"
            kwargs[transBatchName] = 1 - kwargs[transBatchName]  # type: ignore[assignment]
        if self._flip_transpose:
            transName = f"trans{name}"
            kwargs[transName] = 1 - kwargs[transName]  # type: ignore[assignment]
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


TransposeFusedMatMulWithFlippedBatchAndTranspose1 = type(
    "TransposeFusedMatMulWithFlippedBatchAndTranspose1",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_flip_transpose": True, "_flip_transpose_batch": True},
)
TransposeFusedMatMulWithFlippedBatchAndTranspose2 = type(
    "TransposeFusedMatMulWithFlippedBatchAndTranspose2",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_pos": 2, "_flip_transpose": True, "_flip_transpose_batch": True},
)
TransposeFusedMatMulWithFlippedBatch1 = type(
    "TransposeFusedMatMulWithFlippedBatch1",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_flip_transpose_batch": True},
)
TransposeFusedMatMulWithFlippedBatch2 = type(
    "TransposeFusedMatMulWithFlippedBatch2",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_pos": 2, "_flip_transpose_batch": True},
)
TransposeFusedMatMulWithBatchAndTranspose1 = type(
    "TransposeFusedMatMulWithBatchAndTranspose1",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_flip_transpose": True},
)
TransposeFusedMatMulWithBatchAndTranspose2 = type(
    "TransposeFusedMatMulWithBatchAndTranspose2",
    (_TransposeFusedMatMulBaseWithBatch,),
    {"_pos": 2, "_flip_transpose": True},
)


class MatMulTranspose(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Transpose`` with FusedMatMul."""

    def pattern(self, op, x, y):
        return op.Transpose(op.MatMul(x, y), _outputs=["transposed"])

    def check(self, context, x, y, transposed: ir.Value, **_) -> orp.MatchResult:
        check_result = orp.MatchResult()
        transpose_node = get_node(transposed, "Transpose")
        perm = transpose_node.attributes["perm"].as_ints()
        expected_perm = list(range(len(perm)))
        expected_perm[-2], expected_perm[-1] = expected_perm[-1], expected_perm[-2]
        if perm != expected_perm:
            return check_result.fail("Permutation values for Transpose are not correct.")
        return check_result

    def rewrite(self, op, x, y, fused: ir.Value | None = None, **_):
        kwargs = {}
        if fused:
            fused_node = get_node(fused, "FusedMatMul")
            kwargs = get_kwargs(fused_node)
        for name in ["transA", "transB"]:
            kwargs[name] = 1 - kwargs.get(name, 0)  # type: ignore[assignment]
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
            TransposeFusedMatMulWithFlippedBatch1.rule(),  # type: ignore[attr-defined]
            TransposeFusedMatMulWithFlippedBatch2.rule(),  # type: ignore[attr-defined]
            TransposeFusedMatMulWithFlippedBatchAndTranspose1.rule(),  # type: ignore[attr-defined]
            TransposeFusedMatMulWithFlippedBatchAndTranspose2.rule(),  # type: ignore[attr-defined]
            TransposeFusedMatMulWithBatchAndTranspose1.rule(),  # type: ignore[attr-defined]
            TransposeFusedMatMulWithBatchAndTranspose2.rule(),  # type: ignore[attr-defined]
        ]
    )
