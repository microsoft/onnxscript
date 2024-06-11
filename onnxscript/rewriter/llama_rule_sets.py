# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnx.numpy_helper

import onnxscript.ir as ir
import onnxscript.rewriter.no_op as no_op
import onnxscript.rewriter.pattern as orp

op = orp.onnxop


class CastIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Cast(., to=to)`` by ``Identity`` if possible."""

    @classmethod
    def pattern(cls, op, x, to: int):
        return op.Cast(x, to=to)

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: int):
        return op.Identity(x)

    @classmethod
    def check(cls, context, x, to: int) -> bool:
        return x.dtype == to.value


class CastCast(orp.RewriteRuleAsClass):
    """Replaces ``Cast(Cast(X, ...), to=to)`` by ``Cast(X, to=to)``."""

    @classmethod
    def pattern(cls, op, x, to: int, to0: int):
        return op.Cast(op.Cast(x, to=to0), to=to)

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: int, to0: int):
        return op.Cast(x, to=to)


class ExpandIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Expand(., shape)`` by ``Identity`` if possible."""

    @classmethod
    def pattern(cls, op, x, shape):
        return op.Expand(x, shape)

    @classmethod
    def rewrite(cls, op, x: ir.Value, shape: ir.Value):
        return op.Identity(x)

    @classmethod
    def check(cls, context, x, shape) -> bool:
        shape_x = x.shape
        return shape_x.dims == tuple(shape.const_value.numpy().tolist())


class ReshapeReshape(orp.RewriteRuleAsClass):
    """Replaces ``Reshape(Reshape(X, ...), shape)`` by ``Reshape(X, shape)``."""

    @classmethod
    def pattern(cls, op, x, shape_ignored, shape):
        return op.Reshape(op.Reshape(x, shape_ignored), shape)

    @classmethod
    def rewrite(cls, op, x: ir.Value, shape_ignored: ir.Value, shape: ir.Value):
        return op.Reshape(x, shape)


class TransposeIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(. perm=perm)``
    when the permutation is identity.
    """

    @classmethod
    def pattern(cls, op, x, perm):
        return op.Transpose(x, perm=perm)

    @classmethod
    def check(cls, context, x: ir.Value, perm: ir.Attr | ir.RefAttr) -> bool:
        if isinstance(perm, ir.RefAttr):
            return False
        if perm.type == ir.AttributeType.INTS:
            if perm.value == list(range(len(perm.value))):
                return True
        return False

    @classmethod
    def rewrite(cls, op, x: ir.Value, perm: ir.Attr | None = None):
        return op.Identity(x)


class TransposeTranspose(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(Transpose(., perm=perm1), perm=perm2)``
    when both permutations are inverse.
    """

    @classmethod
    def pattern(cls, op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    @classmethod
    def check(
        cls, context, x: ir.Value, perm1: ir.Attr | ir.RefAttr, perm2: ir.Attr | ir.RefAttr
    ) -> bool:
        if isinstance(perm1, ir.RefAttr) or isinstance(perm2, ir.RefAttr):
            return False
        return True

    @classmethod
    def _apply_transpose(cls, perm: tuple[int, ...], on: list[int]) -> list[int]:
        assert len(perm) == len(on), "length mismatch"
        res = [-1 for i in on]
        for i, p in enumerate(perm):
            res[i] = on[p]
        return res

    @classmethod
    def _apply_transposes(
        cls, perms: list[tuple[int, ...]], on: list[int] | None = None
    ) -> list[int]:
        if on is None:
            on = list(range(len(perms[0])))
        for p in perms:
            on = cls._apply_transpose(p, on)
        return on

    @classmethod
    def rewrite(cls, op, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr):
        first = list(range(len(perm1.value)))
        last = cls._apply_transposes([perm1.value, perm2.value])
        if first == last:
            return op.Identity(x)
        return op.Transpose(x, perm=last)


class UnsqueezeUnsqueeze(orp.RewriteRuleAsClass):
    """Replaces ``Unsqueeze(Unsqueeze(., axes1), axes2)``
    with one Unsqueeze.
    """

    @classmethod
    def pattern(cls, op, x, axes1, axes2):
        return op.Unsqueeze(op.Unsqueeze(x, axes1), axes2)

    @classmethod
    def _combine1(cls, axes1: np.ndarray, axes2: np.ndarray) -> np.ndarray:
        if axes1[0] < axes2[0]:
            return np.hstack([axes1, axes2])
        return np.hstack([axes2, axes1 + 1]).astype(np.int64)

    @classmethod
    def rewrite(cls, op, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        v1 = axes1.const_value.numpy()
        v2 = axes2.const_value.numpy()
        if len(v1) != 1 or len(v2) != 1:
            # Implemented later if needed.
            return False
        axes = cls._combine1(v1, v2)
        return op.Unsqueeze(x, op.Constant(value=onnx.numpy_helper.from_array(axes)))


cast_cast_rule = orp.make_rewrite_rule_from_class(CastCast)
cast_identity_rule = orp.make_rewrite_rule_from_class(CastIdentity)
expand_identity_rule = orp.make_rewrite_rule_from_class(ExpandIdentity)
reshape_reshape_rule = orp.make_rewrite_rule_from_class(ReshapeReshape)
transpose_identity_rule = orp.make_rewrite_rule_from_class(TransposeIdentity)
transpose_transpose_rule = orp.make_rewrite_rule_from_class(TransposeTranspose)
unsqueeze_unsqueeze_rule = orp.make_rewrite_rule_from_class(UnsqueezeUnsqueeze)


def llama_p0_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before any other one as they usually remove unnecessary computation
    such as the multiplication by 1 or two consecutive transpose.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            no_op.mul_by_1_rule,
            no_op.add_0_rule,
            no_op.add_0_rule,
            no_op.div_by_1_rule,
            cast_cast_rule,
            cast_identity_rule,
            expand_identity_rule,
            reshape_reshape_rule,
            transpose_identity_rule,
            transpose_transpose_rule,
            unsqueeze_unsqueeze_rule,
        ]
    )
