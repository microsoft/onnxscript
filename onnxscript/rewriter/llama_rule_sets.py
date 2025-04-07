# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import ClassVar

import onnx.numpy_helper

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


class CastIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Cast(., to=to)`` by ``Identity`` if possible."""

    @classmethod
    def pattern(cls, op, x, to):
        return op.Cast(x, to=to)

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: ir.Attr):
        return op.Identity(x)

    @classmethod
    def check(cls, context, x, to) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if x.dtype != to.value:
            return check_result.fail("Input and output types are not the same")
        return check_result


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
    def check(cls, context, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if to.value not in cls._allowed_tensor_types:
            return check_result.fail(f"Output type {to.value} is not allowed")
        if to_ignored.value not in cls._allowed_tensor_types:
            return check_result.fail(f"Ignored type {to_ignored.value} is not allowed")
        return check_result

    @classmethod
    def rewrite(cls, op, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr):
        return op.Cast(x, to=to)


class ExpandIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Expand(..., shape)`` by ``Identity`` if possible."""

    @classmethod
    def pattern(cls, op, x, shape):
        return op.Expand(x, shape)

    @classmethod
    def rewrite(cls, op, x: ir.Value, shape: ir.Value):
        return op.Identity(x)

    @classmethod
    def check(cls, context, x, shape) -> orp.MatchResult:
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
    def check(cls, context, x, shape_ignored, shape) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if shape_ignored.const_value is None:
            return check_result.fail("Shape ignored is not a constant.")
        if shape.const_value is None:
            return check_result.fail("Shape is not a constant.")
        if shape.const_value.numpy().min() <= 0:
            return check_result.fail("Shape has non-positive values.")
        return check_result


class SlicesSplit(orp.RewriteRuleAsClass):
    """Replaces ``Slice(x, ...), Slice(x, ...)``
    by ``Split(x, ...)`` if possible.
    """

    @classmethod
    def pattern(cls, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Slice(x, begin0, end0, axes0), op.Slice(x, begin1, end1, axes1)

    @classmethod
    def check(cls, context, x, begin0, end0, axes0, begin1, end1, axes1) -> orp.MatchResult:
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

    @classmethod
    def rewrite(cls, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Split(x, num_outputs=2, axis=-1, _outputs=2)


class TransposeIdentity(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(. perm=perm)``
    when the permutation is identity.
    """

    @classmethod
    def pattern(cls, op, x, perm):
        return op.Transpose(x, perm=perm)

    @classmethod
    def check(cls, context, x: ir.Value, perm: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if isinstance(perm, ir.RefAttr):
            return check_result.fail("Permutation is a reference attribute.")
        if perm.type == ir.AttributeType.INTS:
            if perm.value == list(range(len(perm.value))):
                return check_result
        return check_result.fail("Permutation is not identity.")

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
    def check(cls, context, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr) -> orp.MatchResult:
        check_result = orp.MatchResult()
        if isinstance(perm1, ir.RefAttr) or isinstance(perm2, ir.RefAttr):
            return check_result.fail("Permutation is a reference attribute.")
        return check_result

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
    """Replaces ``Unsqueeze(Unsqueeze(., axes1), axes2)`` with one Unsqueeze."""

    @classmethod
    def pattern(cls, op, x, axes1, axes2):
        return op.Unsqueeze(op.Unsqueeze(x, axes1), axes2)

    @classmethod
    def rewrite(cls, op, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        v1 = ir_utils.get_singleton_value(axes1)
        v2 = ir_utils.get_singleton_value(axes2)
        axes = [v1, v2] if v1 < v2 else [v2, v1 + 1]
        return op.Unsqueeze(x, op.Constant(value=ir.tensor(axes, dtype=ir.DataType.INT64)))

    @classmethod
    def check(cls, context, x, axes1, axes2) -> orp.MatchResult:
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


cast_cast_rule = orp.make_rewrite_rule_from_class(CastCast)
cast_identity_rule = orp.make_rewrite_rule_from_class(CastIdentity)
expand_identity_rule = orp.make_rewrite_rule_from_class(ExpandIdentity)
reshape_reshape_rule = orp.make_rewrite_rule_from_class(ReshapeReshape)
slice_split_rule = orp.make_rewrite_rule_from_class(SlicesSplit, True)
transpose_identity_rule = orp.make_rewrite_rule_from_class(TransposeIdentity)
transpose_transpose_rule = orp.make_rewrite_rule_from_class(TransposeTranspose)
unsqueeze_unsqueeze_rule = orp.make_rewrite_rule_from_class(UnsqueezeUnsqueeze)
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
