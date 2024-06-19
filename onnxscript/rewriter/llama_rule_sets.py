# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import numpy as np
import onnx.numpy_helper

import onnxscript.ir as ir
import onnxscript.rewriter.no_op as no_op
import onnxscript.rewriter.pattern as orp

op = orp.onnxop


class CastIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Cast(., to=to)`` by ``Identity`` if possible."""

    @classmethod
    def pattern(cls, op, x, to):
        return op.Cast(x, to=to)

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: ir.AttrInt64):
        return op.Identity(x)

    @classmethod
    def check(cls, context, x, to) -> bool:
        return x.dtype == to.value


class CastCast(orp.RewriteRuleAsClass):
    """Replaces ``Cast(Cast(X, ...), to=to)`` by ``Cast(X, to=to)``."""

    _allowed_tensor_types: ClassVar = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.DOUBLE,
    }

    @classmethod
    def pattern(cls, op, x, to, to_ignored):
        return op.Cast(op.Cast(x, to=to_ignored), to=to)

    @classmethod
    def check(cls, context, x: ir.Value, to: ir.AttrInt64, to_ignored: ir.AttrInt64) -> bool:
        return (
            to.value in cls._allowed_tensor_types
            and to_ignored.value in cls._allowed_tensor_types
        )

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: ir.AttrInt64, to_ignored: ir.AttrInt64):
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
        if shape.const_value is None:
            # Shape is not a constant and cannot be guessed.
            return False
        shape_x = x.shape
        return shape_x.dims == tuple(shape.const_value.numpy().tolist())


class ReshapeReshape(orp.RewriteRuleAsClass):
    """Replaces ``Reshape(Reshape(X, ...), shape)`` by ``Reshape(X, shape)``.
    The pattern matches only if second reshape reshapes into a shape
    with positive values.
    """

    @classmethod
    def pattern(cls, op, x, shape_ignored, shape):
        return op.Reshape(op.Reshape(x, shape_ignored), shape)

    @classmethod
    def rewrite(cls, op, x: ir.Value, shape_ignored: ir.Value, shape: ir.Value):
        return op.Reshape(x, shape)

    @classmethod
    def check(cls, context, x, shape_ignored, shape) -> bool:
        if shape_ignored.const_value is None or shape.const_value is None:
            return False
        if shape.const_value.numpy().min() <= 0:
            return False
        return True


class SlicesSplit(orp.RewriteRuleAsClass):
    """Replaces ``Slice(x, ...), Slice(x, ...)``
    by ``Split(x, ...)`` if possible.
    """

    @classmethod
    def pattern(cls, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Slice(x, begin0, end0, axes0), op.Slice(x, begin1, end1, axes1)

    @classmethod
    def check(cls, context, x, begin0, end0, axes0, begin1, end1, axes1) -> bool:
        if (
            axes0.const_value is None
            or axes1.const_value is None
            or axes0.const_value.numpy().tolist() != axes1.const_value.numpy().tolist()
        ):
            return False
        axes = axes0.const_value.numpy().tolist()
        if len(axes) != 1:
            return False
        if x.shape:
            rk = len(x.shape)
        else:
            rk = x.rank
        if axes[0] != -1 and axes[0] != rk - 1:
            return False
        if (
            begin0.const_value is None
            or end0.const_value is None
            or begin1.const_value is None
            or end1.const_value is None
        ):
            return False
        if begin0.const_value.numpy().tolist() != [0]:
            return False
        e0, b1, e1 = (
            end0.const_value.numpy().tolist(),
            begin1.const_value.numpy().tolist(),
            end1.const_value.numpy().tolist(),
        )
        if e0[0] != b1[0]:
            return False
        shape = x.shape
        if shape is None:
            return False
        last_dim = shape[-1]
        if not isinstance(last_dim, int):
            return False
        if last_dim != e1[0]:
            return False
        if last_dim // 2 != b1[0]:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Split(x, num_outputs=2, axis=-1, outputs=2)


class TransposeIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(. perm=perm)``
    when the permutation is identity.
    """

    @classmethod
    def pattern(cls, op, x, perm):
        return op.Transpose(x, perm=perm)

    @classmethod
    def check(cls, context, x: ir.Value, perm: ir.Attr) -> bool:
        if isinstance(perm, ir.RefAttr):
            return False
        if perm.type == ir.AttributeType.INTS:
            if perm.value == list(range(len(perm.value))):
                return True
        return False

    @classmethod
    def rewrite(cls, op, x: ir.Value, perm: ir.Attr):
        return op.Identity(x)


class TransposeTranspose(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(Transpose(., perm=perm1), perm=perm2)``
    when both permutations are inverse.
    """

    @classmethod
    def pattern(cls, op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    @classmethod
    def check(cls, context, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr) -> bool:
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
    def _combine_axes(cls, axes1: np.ndarray, axes2: np.ndarray) -> np.ndarray:
        """Combines two single axes into one tensor of two axes."""
        if axes1[0] < axes2[0]:
            return np.hstack([axes1, axes2])
        return np.hstack([axes2, axes1 + 1]).astype(np.int64)

    @classmethod
    def rewrite(cls, op, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        v1 = axes1.const_value.numpy()  # type: ignore[union-attr]
        v2 = axes2.const_value.numpy()  # type: ignore[union-attr]
        if len(v1) != 1 or len(v2) != 1:
            # Implemented later if needed.
            return False
        axes = cls._combine_axes(v1, v2)
        return op.Unsqueeze(x, op.Constant(value=onnx.numpy_helper.from_array(axes)))

    @classmethod
    def check(cls, context, x, axes1, axes2) -> bool:
        if axes1.const_value is None or axes2.const_value is None:
            return False
        if axes1.const_value.numpy().min() < 0:
            return False
        if axes2.const_value.numpy().min() < 0:
            return False
        return True


def llama_p0_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before any other one as they usually remove unnecessary computation
    such as the multiplication by 1 or two consecutive transpose.

    Returns:
        RewriteRuleSet
    """
    cast_cast_rule = orp.make_rewrite_rule_from_class(CastCast)
    cast_identity_rule = orp.make_rewrite_rule_from_class(CastIdentity)
    expand_identity_rule = orp.make_rewrite_rule_from_class(ExpandIdentity)
    reshape_reshape_rule = orp.make_rewrite_rule_from_class(ReshapeReshape)
    slice_split_rule = orp.make_rewrite_rule_from_class(SlicesSplit, True)
    transpose_identity_rule = orp.make_rewrite_rule_from_class(TransposeIdentity)
    transpose_transpose_rule = orp.make_rewrite_rule_from_class(TransposeTranspose)
    unsqueeze_unsqueeze_rule = orp.make_rewrite_rule_from_class(UnsqueezeUnsqueeze)

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
            slice_split_rule,
            transpose_identity_rule,
            transpose_transpose_rule,
            unsqueeze_unsqueeze_rule,
        ]
    )
