# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Basic rewrite rules for general optimization patterns.

This module contains fundamental optimization rules that are generally applicable
to most ONNX models, including cast elimination, transpose simplification,
shape operation fusion, and other common patterns.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from onnxscript import ir
from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class SqueezeReshape(RewriteRuleClassBase):
    """Replaces ``Reshape(Squeeze(x), [-1]])`` with ``Identity(x)`` for 1D x.

    This pattern arises from the translation of pytorch symints.
    """

    def __init__(self):
        super().__init__("SqueezeReshape1d", remove_nodes=False)

    def pattern(self, op, x):
        return op.Reshape(op.Squeeze(x), [-1])

    def rewrite(self, op, x: ir.Value):
        return op.Identity(x)

    def check(self, context, x) -> MatchResult:
        del context  # Unused
        check_result = MatchResult()
        if not ir_utils.has_rank(x, 1):
            return check_result.fail("Input is not 1D")
        return check_result


class CastIdentity(RewriteRuleClassBase):
    """Replaces ``Cast(., to=to)`` by ``Identity`` if possible."""

    def pattern(self, op, x, to):
        return op.Cast(x, to=to)

    def rewrite(self, op, x: ir.Value, to: ir.Attr):
        return op.Identity(x)

    def check(self, context, x, to) -> MatchResult:
        check_result = MatchResult()
        if x.dtype != to.as_int():
            return check_result.fail("Input and output types are not the same")
        return check_result


class CastCast(RewriteRuleClassBase):
    """Replaces ``Cast(Cast(X, ...), to=to)`` by ``Cast(X, to=to)``."""

    # Simplify "cast type1 => type2 => type3" to "cast type1 => type3".
    # This rule is not valid for all combinations of types: e.g.,
    # it is not valid for float32 => float16 => float32 or float32 => int32 => string.
    # TODO: fill out the list of allowed combinations: the following is just a couple
    # that shows up in practice where it is valid
    _allowed_type2_type3: ClassVar = frozenset(
        {
            (ir.DataType.FLOAT, ir.DataType.FLOAT16),
            (ir.DataType.FLOAT, ir.DataType.BFLOAT16),
        }
    )

    def pattern(self, op, x, to, to_ignored):
        return op.Cast(op.Cast(x, to=to_ignored), to=to)

    def check(self, context, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr) -> MatchResult:
        check_result = MatchResult()
        type2 = to_ignored.as_int()
        type3 = to.as_int()
        if (type2, type3) not in self._allowed_type2_type3:
            return check_result.fail(
                f"Intermediate cast elimination not recognized as valid from {type2} to {type3}. "
                f"Cast-Cast rule may be incomplete for this combination."
            )
        return check_result

    def rewrite(self, op, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr):
        return op.Cast(x, to=to)


class ExpandIdentity(RewriteRuleClassBase):
    """Replaces ``Expand(..., shape)`` by ``Identity`` if possible."""

    def pattern(self, op, x, shape):
        return op.Expand(x, shape)

    def rewrite(self, op, x: ir.Value, shape: ir.Value):
        return op.Identity(x)

    def check(self, context, x, shape) -> MatchResult:
        check_result = MatchResult()
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


class ReshapeReshape(RewriteRuleClassBase):
    """Replaces ``Reshape(Reshape(X, ...), shape)`` by ``Reshape(X, shape)``.
    The pattern matches only if second reshape reshapes into a shape
    with positive values.
    """

    def pattern(self, op, x, shape_ignored, shape):
        return op.Reshape(op.Reshape(x, shape_ignored), shape)

    def rewrite(self, op, x: ir.Value, shape_ignored: ir.Value, shape: ir.Value):
        new_shape = op.initializer(ir.Tensor(self._new_shape, name=shape.name))
        return op.Reshape(x, new_shape, allowzero=self._allowzero)

    def check(self, context, x, shape_ignored, shape) -> MatchResult:
        check_result = MatchResult()

        # Shape must be a constant.
        if (np_shape := ir_utils.get_numpy_value(shape)) is None:
            return check_result.fail("Shape is not a constant.")
        # Convert to array to support assignment destination.
        self._new_shape = np.array(np_shape, np_shape.dtype)

        # Try to replace {0,-1} values in shape if reshape output is known.
        if (reshape_output := context.output_values[0].shape) is not None:
            for i, dim in enumerate(reshape_output):
                if isinstance(dim, int) and dim > 0:
                    self._new_shape[i] = dim

        # Constraints for shape.
        self._allowzero = context.nodes[0].attributes.get_int("allowzero", 0)
        if self._allowzero == 1 and any(self._new_shape == 0):
            return check_result
        if any(self._new_shape == 0) and any(self._new_shape < 0):
            return check_result.fail("Shape cannot contain both 0 and -1 dimensions.")
        elif np.count_nonzero(self._new_shape == 0) > 1:
            return check_result.fail("Shape cannot contain more than one 0 dimension.")

        # At this point, we can safely replace '0' with '-1'.
        # Note allowzero is removed since at this point it does not have any effect.
        self._allowzero = None
        self._new_shape = np.where(self._new_shape == 0, -1, self._new_shape)
        return check_result


class SlicesSplit(RewriteRuleClassBase):
    """Replaces ``Slice(x, ...), Slice(x, ...)``
    by ``Split(x, ...)`` if possible.
    """

    def pattern(self, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Slice(x, begin0, end0, axes0), op.Slice(x, begin1, end1, axes1)

    def check(self, context, x, begin0, end0, axes0, begin1, end1, axes1) -> MatchResult:
        check_result = MatchResult()
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


class TransposeIdentity(RewriteRuleClassBase):
    """Replaces ``Transpose(. perm=perm)``
    when the permutation is identity.
    """

    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, context, x: ir.Value, perm: ir.Attr) -> MatchResult:
        check_result = MatchResult()
        if perm.is_ref():
            return check_result.fail("Permutation is a reference attribute.")
        if perm.type == ir.AttributeType.INTS:
            perm_ints = tuple(perm.as_ints())
            if perm_ints == tuple(range(len(perm_ints))):
                return check_result
        return check_result.fail("Permutation is not identity.")

    def rewrite(self, op, x: ir.Value, perm: ir.Attr):
        return op.Identity(x)


class TransposeTranspose(RewriteRuleClassBase):
    """Replaces ``Transpose(Transpose(., perm=perm1), perm=perm2)``
    when both permutations are inverse.
    """

    def pattern(self, op, x, perm1, perm2):
        return op.Transpose(op.Transpose(x, perm=perm1), perm=perm2)

    def check(self, context, x: ir.Value, perm1: ir.Attr, perm2: ir.Attr) -> MatchResult:
        check_result = MatchResult()
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


class UnsqueezeUnsqueeze(RewriteRuleClassBase):
    """Replaces ``Unsqueeze(Unsqueeze(., axes1), axes2)`` with one Unsqueeze."""

    def pattern(self, op, x, axes1, axes2):
        return op.Unsqueeze(op.Unsqueeze(x, axes1), axes2)

    def rewrite(self, op, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        v1 = ir_utils.get_singleton_value(axes1)
        v2 = ir_utils.get_singleton_value(axes2)
        axes = [v1, v2] if v1 < v2 else [v2, v1 + 1]
        return op.Unsqueeze(x, op.Constant(value=ir.tensor(axes, dtype=ir.DataType.INT64)))

    def check(self, context, x, axes1, axes2) -> MatchResult:
        check_result = MatchResult()
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


class Flatten2Reshape(RewriteRuleClassBase):
    """Convert ``Flatten(x)`` to Reshape."""

    def pattern(self, op, x: ir.Value):
        return op.Flatten(x)

    def rewrite(self, op, x: ir.Value):
        new_shape = op.initializer(ir.Tensor(self._new_shape, name=f"{x.name}/shape"))
        return op.Reshape(x, new_shape)

    def check(self, context, x: ir.Value) -> MatchResult:
        check_result = MatchResult()
        self._new_shape = np.array([-1, -1], "int64")

        # Convert axis in a positive value if possible.
        axis = context.root.attributes.get_int("axis", 1)
        input_rank = None
        if (input_shape := x.shape) is not None:
            input_rank = len(input_shape)
            if axis < 0:
                axis += input_rank

        # Compute reshape shape following axis attribute.
        if axis == 0:
            self._new_shape[0] = 1
        elif axis == 1:
            self._new_shape[0] = 0
        elif axis == input_rank:
            self._new_shape[1] = 1

        # Try to update shape if output is known.
        if (output_shape := context.output_values[0].shape) is not None:
            for i, dim in enumerate(output_shape):
                if isinstance(dim, int):
                    self._new_shape[i] = dim

        # Try to update shape if input is known.
        if input_shape is not None:
            if all(isinstance(dim, int) for dim in input_shape[:axis]):
                self._new_shape[0] = np.prod(input_shape[:axis])
            if all(isinstance(dim, int) for dim in input_shape[axis:]):
                self._new_shape[1] = np.prod(input_shape[axis:])

        # Verify if it is possible to apply rule.
        if np.count_nonzero(self._new_shape == -1) > 1:
            return check_result.fail("Impossible to compute new shape.")
        return check_result


# Create rule instances
cast_cast_rule = CastCast.rule()
no_op_cast_rule = CastIdentity.rule()
no_op_expand_rule = ExpandIdentity.rule()
reshape_reshape_rule = ReshapeReshape.rule()
slice_split_rule = SlicesSplit.rule()
no_op_transpose_rule = TransposeIdentity.rule()
transpose_transpose_rule = TransposeTranspose.rule()
unsqueeze_unsqueeze_rule = UnsqueezeUnsqueeze.rule()
squeeze_reshape_1d_rule = SqueezeReshape.rule()
flatten_to_reshape_rule = Flatten2Reshape.rule()


def basic_optimization_rules() -> RewriteRuleSet:
    """Returns a set of basic optimization rules.

    These rules perform fundamental optimizations such as:
    - Eliminating redundant cast operations
    - Simplifying consecutive operations of the same type
    - Removing identity operations
    - Optimizing shape manipulation operations

    These rules are generally safe to apply as a first optimization pass
    before other more specialized optimizations.

    Returns:
        RewriteRuleSet: A collection of basic optimization rules
    """
    return RewriteRuleSet(
        [
            cast_cast_rule,
            no_op_cast_rule,
            no_op_expand_rule,
            # flatten_to_reshape_rule is order sensitive to reshape_reshape_rule
            flatten_to_reshape_rule,
            reshape_reshape_rule,
            slice_split_rule,
            no_op_transpose_rule,
            transpose_transpose_rule,
            unsqueeze_unsqueeze_rule,
            squeeze_reshape_1d_rule,
        ]
    )
