# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import onnx_ir as ir

import onnxscript.rewriter.pattern as orp
from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchFailureError


def _get_node(value: ir.Value, name: str) -> ir.Node:
    """Get the node from the output value."""
    node = value.producer()
    assert node is not None, f"{name} node should not be None"
    return node


def _get_kwargs(node: ir.Node) -> dict[str, float | int]:
    """Get the kwargs from the node."""
    kwargs = {key: val.value for key, val in node.attributes.items()}
    return kwargs


def _division_alpha(x: ir.Value, y: ir.Value, cst: ir.Value, alpha: float = 1.0) -> float:
    """Validate division scaling before encoding it in ORT's float32 alpha attribute.

    Nonfinite divisors are intentionally left as Div: replacing division by infinity
    with multiplication by zero changes the behavior of infinite matrix products.
    Ordinary floating-point reciprocal rounding is allowed, but overflow/underflow
    is not. Subnormal scales are declined to avoid lost relative precision and
    provider-specific flushing to zero. Double precision requires an exact alpha.
    """
    supported = {
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
        ir.DataType.BFLOAT16,
    }
    if x.dtype not in supported or y.dtype != x.dtype or cst.dtype != x.dtype:
        raise MatchFailureError(
            "FusedMatMul division requires matching supported floating-point dtypes.", cst
        )
    if cst.const_value is None:
        raise MatchFailureError("Divisor is not a constant value.", cst)
    c = _ir_utils.get_singleton_value(cst)
    if c is None:
        raise MatchFailureError("Divisor is not a singleton value.", cst)
    c = float(c)
    if not math.isfinite(c) or c == 0:
        raise MatchFailureError("Divisor must be finite and nonzero.", cst)
    if not math.isfinite(alpha):
        raise MatchFailureError("Existing FusedMatMul alpha must be finite.")
    scaled_alpha = alpha / c
    # ONNX FLOAT attributes are float32 even when the tensor dtype is double.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        attribute_alpha = float(np.float32(scaled_alpha))
        tensor_alpha = float(np.asarray(attribute_alpha, dtype=x.dtype.numpy()))
    # Bfloat16 has the same normal exponent range as float32; double is bounded
    # by the float32 attribute. Float16 has a narrower normal range.
    min_normal = np.finfo(np.float16 if x.dtype == ir.DataType.FLOAT16 else np.float32).tiny
    if (
        not math.isfinite(attribute_alpha)
        or not math.isfinite(tensor_alpha)
        or (
            alpha != 0
            and (abs(attribute_alpha) < min_normal or abs(tensor_alpha) < min_normal)
        )
    ):
        raise MatchFailureError(
            "Division alpha overflows or underflows the FusedMatMul attribute or tensor dtype.",
            cst,
        )
    if x.dtype == ir.DataType.DOUBLE and attribute_alpha != scaled_alpha:
        raise MatchFailureError("Division alpha is not exactly representable as float32.", cst)
    return attribute_alpha


class FusedMatMulDiv1(orp.RewriteRuleClassBase):
    """Replaces ``MatMul + Div`` with MatMul."""

    def pattern(self, op, x, y, cst):
        return op.Div(op.MatMul(x, y), cst)

    def check(self, context, x, y, cst) -> orp.MatchResult:
        _division_alpha(x, y, cst)
        return orp.MatchResult()

    def rewrite(self, op, x, y, cst):
        return op.FusedMatMul(x, y, alpha=_division_alpha(x, y, cst), _domain="com.microsoft")


class FusedMatMulDiv2(orp.RewriteRuleClassBase):
    """Replaces ``FusedMatMul + Div`` with FusedMatMul."""

    def pattern(self, op, x, y, cst):
        return op.Div(op.FusedMatMul(x, y, _domain="com.microsoft", _outputs=["fused"]), cst)

    def check(self, context, x, y, cst, fused: ir.Value, **_) -> orp.MatchResult:
        alpha = _get_node(fused, "FusedMatMul").attributes.get_float("alpha", 1.0)
        _division_alpha(x, y, cst, alpha)
        return orp.MatchResult()

    def rewrite(self, op, x, y, cst, fused: ir.Value):
        fused_node = _get_node(fused, "FusedMatMul")
        kwargs = _get_kwargs(fused_node)
        kwargs["alpha"] = _division_alpha(x, y, cst, kwargs.get("alpha", 1.0))
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
