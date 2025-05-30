# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar, Sequence

from onnxscript import ir
from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter import pattern as orp


class SqueezeReshape(orp.RewriteRuleClassBase):
    """Replaces ``Reshape(Squeeze(x), [-1]])`` with ``Identity(x)`` for 1D x.

    This pattern arises from the translation of pytorch symints.
    """

    def __init__(self):
        super().__init__("SqueezeReshape1d", remove_nodes=False)

    def pattern(self, op, x):
        return op.Reshape(op.Squeeze(x), [-1])

    def rewrite(self, op, x: ir.Value):
        return op.Identity(x)

    def check(self, context, x) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()
        if not ir_utils.has_rank(x, 1):
            return check_result.fail("Input is not 1D")
        return check_result


class CastIdentity(orp.RewriteRuleClassBase):
    """Replaces ``Cast(., to=to)`` by ``Identity`` if possible."""

    def pattern(self, op, x, to):
        return op.Cast(x, to=to)

    def rewrite(self, op, x: ir.Value, to: ir.Attr):
        return op.Identity(x)

    def check(self, context, x, to) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if x.dtype != to.as_int():
            return check_result.fail("Input and output types are not the same")
        return check_result


class CastCast(orp.RewriteRuleClassBase):
    """Replaces ``Cast(Cast(X, ...), to=to)`` by ``Cast(X, to=to)``."""

    _allowed_tensor_types: ClassVar = {
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.DOUBLE,
    }

    def pattern(self, op, x, to, to_ignored):
        return op.Cast(op.Cast(x, to=to_ignored), to=to)

    def check(self, context, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if to.as_int() not in self._allowed_tensor_types:
            return check_result.fail(f"Output type {to.as_int()} is not allowed")
        if to_ignored.as_int() not in self._allowed_tensor_types:
            return check_result.fail(f"Ignored type {to_ignored.as_int()} is not allowed")
        return check_result

    def rewrite(self, op, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr):
        return op.Cast(x, to=to)


class ExpandIdentity(orp.RewriteRuleClassBase):
    """Replaces ``Expand(..., shape)`` by ``Identity`` if possible."""

    def pattern(self, op, x, shape):
        return op.Expand(x, shape)

    def rewrite(self, op, x: ir.Value, shape: ir.Value):
        return op.Identity(x)

    def check(self, context, x, shape) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if shape.const_value is None:
            # Shape is not a constant and cannot be guessed.
            return check_result.fail("Shape is not a constant and cannot be guessed.")
        if (x_shape := x.shape) is None:
            # We don't know the shape of the input
            return check_result.fail("Input shape is not known.")
        if x_shape.dims != tuple(shape.const_value.numpy().tolist()):
            return check_result.fail(
                f"Input shape {x_shape.dims} does not match the shape {shape.const_value.numpy().tolist()}."
            )
        return check_result


class ReshapeReshape(orp.RewriteRuleClassBase):
    """Replaces ``Reshape(Reshape(X, ...), shape)`` by ``Reshape(X, shape)``.
    The pattern matches only if second reshape reshapes into a shape
    with positive values.
    """

    def pattern(self, op, x, shape_ignored, shape):
        return op.Reshape(op.Reshape(x, shape_ignored), shape)

    def rewrite(self, op, x: ir.Value, shape_ignored: ir.Value, shape: ir.Value):
        return op.Reshape(x, shape)

    def check(self, context, x, shape_ignored, shape) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if shape_ignored.const_value is None:
            return check_result.fail("Shape ignored is not a constant.")
        if shape.const_value is None:
            return check_result.fail("Shape is not a constant.")
        if shape.const_value.numpy().min() <= 0:
            return check_result.fail("Shape has non-positive values.")
        return check_result


class SlicesSplit(orp.RewriteRuleClassBase):
    """Replaces ``Slice(x, ...), Slice(x, ...)``
    by ``Split(x, ...)`` if possible.
    """

    def pattern(self, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Slice(x, begin0, end0, axes0), op.Slice(x, begin1, end1, axes1)

    def check(self, context, x, begin0, end0, axes0, begin1, end1, axes1) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if (
            axes0.const_value is None
            or axes1.const_value is None
            or axes0.const_value.numpy().tolist() != axes1.const_value.numpy().tolist()
        ):
            return check_result.fail("Axes are not equal or not constant.")
        axes = axes0.const_value.numpy().tolist()
        if len(axes) != 1:
            return check_result.fail("Axes has more than one dimension.")
        if x.shape:
            rk = len(x.shape)
        else:
            rk = x.rank
        if axes[0] != -1 and axes[0] != rk - 1:
            return check_result.fail("Axes is not -1 or last dimension.")
        if (
            begin0.const_value is None
            or end0.const_value is None
            or begin1.const_value is None
            or end1.const_value is None
        ):
            return check_result.fail("Begin or end are not constant values.")
        if begin0.const_value.numpy().tolist() != [0]:
            return check_result.fail("First begin value is not 0.")
        e0, b1, e1 = (
            end0.const_value.numpy().tolist(),
            begin1.const_value.numpy().tolist(),
            end1.const_value.numpy().tolist(),
        )
        if e0[0] != b1[0]:
            return check_result.fail("End0 is not equal to Begin1.")
        shape = x.shape
        if shape is None:
            return check_result.fail("Shape is not known.")
        last_dim = shape[-1]
        if not isinstance(last_dim, int):
            return check_result.fail("Last dimension is not known.")
        if last_dim != e1[0]:
            return check_result.fail("Last dimension is not equal to End1.")
        if last_dim // 2 != b1[0]:
            return check_result.fail("Last dimension is not equal to Begin1.")
        return check_result

    def rewrite(self, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Split(x, num_outputs=2, axis=-1, _outputs=2)


class TransposeIdentity(orp.RewriteRuleClassBase):
    """Replaces ``Transpose(. perm=perm)``
    when the permutation is identity.
    """

    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, context, x: ir.Value, perm: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if perm.is_ref():
            return check_result.fail("Permutation is a reference attribute.")
        if perm.type == ir.AttributeType.INTS:
            perm_ints = perm.as_ints()
            if perm_ints == list(range(len(perm_ints))):
                return check_result
        return check_result.fail("Permutation is not identity.")

    def rewrite(self, op, x: ir.Value, perm: ir.Attr):
        return op.Identity(x)


class TransposeTranspose(orp.RewriteRuleClassBase):
    """Replaces ``Transpose(Transpose(., perm=perm1), perm=perm2)``
    when both permutations are inverse.
    """

    def pattern(self, op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    def check(self, context, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if perm1.is_ref() or perm2.is_ref():
            return check_result.fail("Permutation is a reference attribute.")
        return check_result

    def _apply_transpose(self, perm: Sequence[int], on: list[int]) -> list[int]:
        assert len(perm) == len(on), "length mismatch"
        res = [-1 for i in on]
        for i, p in enumerate(perm):
            res[i] = on[p]
        return res

    def _apply_transposes(
        self, perms: list[Sequence[int]], on: list[int] | None = None
    ) -> list[int]:
        if on is None:
            on = list(range(len(perms[0])))
        for p in perms:
            on = self._apply_transpose(p, on)
        return on

    def rewrite(self, op, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr):
        first = list(range(len(perm1.as_ints())))
        last = self._apply_transposes([perm1.as_ints(), perm2.as_ints()])
        if first == last:
            return op.Identity(x)
        return op.Transpose(x, perm=last)


class UnsqueezeUnsqueeze(orp.RewriteRuleClassBase):
    """Replaces ``Unsqueeze(Unsqueeze(., axes1), axes2)`` with one Unsqueeze."""

    def pattern(self, op, x, axes1, axes2):
        return op.Unsqueeze(op.Unsqueeze(x, axes1), axes2)

    def rewrite(self, op, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        v1 = ir_utils.get_singleton_value(axes1)
        v2 = ir_utils.get_singleton_value(axes2)
        axes = [v1, v2] if v1 < v2 else [v2, v1 + 1]
        return op.Unsqueeze(x, op.Constant(value=ir.tensor(axes, dtype=ir.DataType.INT64)))

    def check(self, context, x, axes1, axes2) -> orp.MatchResult:
        check_result = orp.MatchResult()
        del context  # Unused
        del x  # Unused
        # Currently restricted to single element positive axis
        v1 = ir_utils.get_singleton_value(axes1)
        v2 = ir_utils.get_singleton_value(axes2)
        if v1 is None or v2 is None:
            return check_result.fail("Axes are not constant.")
        if (v1 < 0) or (v2 < 0):
            return check_result.fail("Axes are negative.")
        return check_result


cast_cast_rule = CastCast.rule()
cast_identity_rule = CastIdentity.rule()
expand_identity_rule = ExpandIdentity.rule()
reshape_reshape_rule = ReshapeReshape.rule()
slice_split_rule = SlicesSplit.rule()
transpose_identity_rule = TransposeIdentity.rule()
transpose_transpose_rule = TransposeTranspose.rule()
unsqueeze_unsqueeze_rule = UnsqueezeUnsqueeze.rule()
squeeze_reshape_1d_rule = SqueezeReshape.rule()


def llama_p0_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before any other one as they usually remove unnecessary computation
    such as the multiplication by 1 or two consecutive transpose.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            # cast_cast_rule,  # Might have precision issues.
            cast_identity_rule,
            expand_identity_rule,
            reshape_reshape_rule,
            slice_split_rule,  # Affect collapse slices rules?
            transpose_identity_rule,
            transpose_transpose_rule,
            unsqueeze_unsqueeze_rule,
            squeeze_reshape_1d_rule,
        ]
    )
