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
import onnx_ir as ir

from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter import _matcher, _pattern_ir
from onnxscript.rewriter._basics import MatchingTracer, MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleClassBase, RewriteRuleSet


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
        new_shape = op.initializer(ir.Tensor(self._new_shape, name=self._new_shape_name))
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
        self._new_shape_name = f"{context.output_values[0].name}/shape"
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


def _slice_half(node: ir.Node) -> tuple[ir.Value, int, int] | None:
    """Identify a Slice of the first or second half of a known even last dimension."""
    if (
        node.op_identifier() != ("", "Slice", "")
        or len(node.inputs) != 4
        or len(node.outputs) != 1
    ):
        return None
    x, begin, end, axes = node.inputs
    if x is None or x.shape is None or len(x.shape) == 0:
        return None
    last_dim = x.shape[-1]
    if not isinstance(last_dim, int) or last_dim <= 0 or last_dim % 2:
        return None
    axis = ir_utils.get_singleton_value(axes, rank=1)
    if not isinstance(axis, int) or axis not in (-1, len(x.shape) - 1):
        return None
    start = ir_utils.get_singleton_value(begin, rank=1)
    stop = ir_utils.get_singleton_value(end, rank=1)
    if (start, stop) == (0, last_dim // 2):
        return x, axis, 0
    if (start, stop) == (last_dim // 2, last_dim):
        return x, axis, 1
    return None


class _SlicesSplitMatcher(_matcher.SimplePatternMatcher):
    def __init__(self, pattern: _pattern_ir.GraphPattern) -> None:
        super().__init__(pattern)
        self._indices: list[
            dict[tuple[ir.Value, int], tuple[list[ir.Node], list[ir.Node]]]
        ] = []

    def setup(self) -> None:
        # A stack preserves the enclosing graph's index while rewriting subgraphs.
        self._indices.append({})

    def cleanup(self) -> None:
        self._indices.pop()

    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        *,
        verbose: int = 0,
        remove_nodes: bool = True,
        tracer: MatchingTracer | None = None,
    ) -> MatchResult:
        del model, tracer
        self._graph = (
            graph_or_function.graph
            if isinstance(graph_or_function, ir.Function)
            else graph_or_function
        )
        self._init_match(verbose)
        key = _slice_half(node)
        if key is None:
            return self._match.fail(
                "Expected a Slice of one half of a known, positive, even last dimension.", node
            )
        x, axis, half = key
        # Direct match calls outside a rewrite pass must not retain an index.
        index = self._indices[-1] if self._indices else {}
        if (x, axis) not in index:
            halves: tuple[list[ir.Node], list[ir.Node]] = ([], [])
            for user, input_index in x.uses():
                if input_index == 0 and user.graph is self._graph:
                    user_key = _slice_half(user)
                    if user_key is not None and user_key[1] == axis:
                        halves[user_key[2]].append(user)
            index[x, axis] = halves
        halves = index[x, axis]
        # Include Slice nodes introduced by earlier rewrites in the same pass.
        halves[half].append(node)
        partners = halves[1 - half]
        while partners:
            partner = partners[-1]
            if partner.graph is not self._graph or _slice_half(partner) != (x, axis, 1 - half):
                # Removed or modified candidates are discarded once, not rescanned.
                partners.pop()
                continue
            pair = (node, partner) if half == 0 else (partner, node)
            return self._multi_match(pair, check_removable=remove_nodes)
        return self._match.fail("No complementary Slice of the same input and axis.", node)


class SlicesSplit(RewriteRuleClassBase):
    """Replace complementary last-axis half Slices with a two-output Split (opset 18+).

    Index the eligible consumers once per input/axis and graph pass, rather than
    enumerating pairs of all Slice consumers of a shared input.
    """

    @classmethod
    def rule(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        matcher = _SlicesSplitMatcher(_pattern_ir._to_graph_pattern(instance.pattern))
        return RewriteRule(
            matcher.pattern,
            instance.rewrite,
            instance.check,
            matcher=matcher,
            name=instance.name,
            remove_nodes=instance.remove_nodes,
            graph_pre_visitor=matcher.setup,
            graph_post_visitor=matcher.cleanup,
            as_function=instance.as_function,
        )

    def pattern(self, op, x, begin0, end0, axes0, begin1, end1, axes1):
        return op.Slice(x, begin0, end0, axes0), op.Slice(x, begin1, end1, axes1)

    def check(self, context, **_) -> MatchResult:
        check_result = MatchResult()
        opset = context.graph_or_function.opset_imports.get(
            "", context.model.graph.opset_imports.get("", 0)
        )
        if opset < 18:
            return check_result.fail("Split with num_outputs requires ONNX opset 18 or newer.")
        first, second = (_slice_half(node) for node in context.nodes)
        if first is None or second is None or first[2] != 0 or second != (*first[:2], 1):
            return check_result.fail(
                "Expected complementary half Slices of the same input and axis."
            )
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

    def pattern(self, op, x):
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
reshape_reshape_rule = ReshapeReshape.rule(remove_nodes=False)
slice_split_rule = SlicesSplit.rule()
no_op_transpose_rule = TransposeIdentity.rule()
transpose_transpose_rule = TransposeTranspose.rule(remove_nodes=False)
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
