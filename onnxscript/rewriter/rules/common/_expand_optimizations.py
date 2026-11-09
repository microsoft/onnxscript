# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optimizations that move or simplify ONNX Expand operations."""

from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _ints(value: ir.Value) -> list[int] | None:
    array = _ir_utils.get_numpy_value(value)
    if array is None or array.ndim != 1 or array.dtype.kind not in "iu":
        return None
    return [int(v) for v in array.reshape(-1)]


def _shape(value: ir.Value) -> list | None:
    if value.shape is None or value.shape.rank() is None:
        return None
    dims = list(value.shape)
    if any(isinstance(dim, ir.SymbolicDim) and dim.value is None for dim in dims):
        return None
    return dims


def _normalize_axes(axes: list[int], rank: int) -> list[int] | None:
    result = []
    for axis in axes:
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank or axis in result:
            return None
        result.append(axis)
    return sorted(result)


def _broadcast(left: list, right: list) -> list | None:
    rank = max(len(left), len(right))
    result = []
    for i in range(rank):
        left_dim = left[i - (rank - len(left))] if i >= rank - len(left) else 1
        right_dim = right[i - (rank - len(right))] if i >= rank - len(right) else 1
        if left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1 or _ir_utils.same_dim(left_dim, right_dim):
            result.append(left_dim)
        else:
            return None
    return result


def _broadcasts_to(source: list, target: list) -> bool:
    if len(source) > len(target):
        return False
    aligned = [1] * (len(target) - len(source)) + source
    return all(
        before == 1 or _ir_utils.same_dim(before, after)
        for before, after in zip(aligned, target)
    )


def _expand_preserves_nonempty_evaluation(source: list, target: list) -> bool:
    """Whether moving computation before Expand cannot evaluate discarded elements."""
    if len(source) > len(target):
        return False
    aligned = [1] * (len(target) - len(source)) + source
    for before, after in zip(aligned, target):
        if _ir_utils.same_dim(before, after):
            continue
        if before == 1 and isinstance(after, int) and after > 0:
            continue
        return False
    return True


def _effective_expand_shape(input_shape: list, target: list[int]) -> list[int] | None:
    rank = max(len(input_shape), len(target))
    aligned_input = [1] * (rank - len(input_shape)) + input_shape
    aligned_target = [1] * (rank - len(target)) + target
    result = []
    for before, requested in zip(aligned_input, aligned_target):
        if not isinstance(before, int):
            return None
        if before == requested or before == 1:
            result.append(requested)
        elif requested == 1:
            result.append(before)
        else:
            return None
    return result


def _default_opset(context) -> int:
    imports = context.graph_or_function.opset_imports
    return imports.get(
        "",
        imports.get(
            "ai.onnx",
            context.model.opset_imports.get("", context.model.opset_imports.get("ai.onnx", 0)),
        ),
    )


class MaterializeExpandShape(RewriteRuleClassBase):
    def pattern(self, op, x, shape):
        return op.Expand(x, shape)

    def check(self, context, x: ir.Value, shape: ir.Value):
        result = MatchResult()
        if _ints(shape) is not None:
            return result.fail("Expand target is already constant.")
        input_shape = _shape(x)
        output_shape = _shape(context.output_values[0])
        if (
            input_shape is None
            or output_shape is None
            or len(input_shape) != len(output_shape)
        ):
            return result.fail("Equal known input and output ranks are required.")
        target = []
        for before, after in zip(input_shape, output_shape):
            if before == after:
                target.append(1)
            elif isinstance(before, int) and before == 1 and isinstance(after, int):
                target.append(after)
            else:
                return result.fail(
                    "Changed dimensions are not statically provable broadcasts."
                )
        self._target = target
        return result

    def rewrite(self, op, x, shape):
        target = op.Constant(value=ir.tensor(self._target, dtype=ir.DataType.INT64))
        return op.Expand(x, target)


class ReplaceUnitExpandWithUnsqueeze(RewriteRuleClassBase):
    """Replace an all-one Expand that only inserts unit dimensions."""

    def __init__(self):
        super().__init__("ReplaceUnitExpandWithUnsqueeze")

    def pattern(self, op, x, shape):
        return op.Expand(x, shape)

    def check(self, context, x: ir.Value, shape: ir.Value):
        result = MatchResult()
        if _default_opset(context) < 13:
            return result.fail("Axes-as-input Unsqueeze requires opset 13 or later.")
        target = _ints(shape)
        before = _shape(x)
        after = _shape(context.output_values[0])
        if (
            target is None
            or before is None
            or after is None
            or any(dim != 1 for dim in target)
        ):
            return result.fail("Expand target must be an all-one constant.")
        if len(before) >= len(after):
            return result.fail("Expand must insert at least one unit dimension.")
        axes = []
        before_index = 0
        for after_index, dim in enumerate(after):
            if before_index < len(before) and dim == before[before_index]:
                before_index += 1
            elif dim == 1:
                axes.append(after_index)
            else:
                return result.fail("Shapes do not differ only by inserted unit dimensions.")
        if before_index != len(before):
            return result.fail("Shapes do not differ only by inserted unit dimensions.")
        self._axes = axes
        return result

    def rewrite(self, op, x, shape):
        axes = op.Constant(value=ir.tensor(self._axes, dtype=ir.DataType.INT64))
        return op.Unsqueeze(x, axes)


class _MoveUnaryBeforeExpand(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"Move{op_type}BeforeExpand")
        self._op_type = op_type

    def pattern(self, op, x, shape):
        return getattr(op, self._op_type)(op.Expand(x, shape))

    def check(self, context, x, shape):
        del shape
        result = MatchResult()
        source = _shape(x)
        target = _shape(context.output_values[0])
        if (
            source is None
            or target is None
            or not _expand_preserves_nonempty_evaluation(source, target)
        ):
            return result.fail(
                "Expand must not turn a nonempty computation into an empty tensor."
            )
        return result

    def rewrite(self, op, x, shape):
        return op.Expand(getattr(op, self._op_type)(x), shape)


class MoveCastBeforeExpand(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveCastBeforeExpand")

    def pattern(self, op, x, shape, to):
        return op.Cast(op.Expand(x, shape), to=to)

    def check(self, context, x, shape, to):
        del shape, to
        source = _shape(x)
        target = _shape(context.output_values[0])
        result = MatchResult()
        if (
            source is None
            or target is None
            or not _expand_preserves_nonempty_evaluation(source, target)
        ):
            return result.fail(
                "Expand must not turn a nonempty computation into an empty tensor."
            )
        self._cast_attributes = dict(context.root.attributes)
        return result

    def rewrite(self, op, x, shape, to):
        return op.Expand(op.Cast(x, **self._cast_attributes), shape)


class MoveUnsqueezeBeforeExpand(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveUnsqueezeBeforeExpand")

    def pattern(self, op, x, expand_shape, axes):
        return op.Unsqueeze(op.Expand(x, expand_shape), axes)

    def check(self, context, x, expand_shape, axes):
        result = MatchResult()
        target = _ints(expand_shape)
        inserted = _ints(axes)
        if target is None or inserted is None:
            return result.fail("Expand shape and Unsqueeze axes must be constants.")
        input_shape = _shape(x)
        if input_shape is None:
            return result.fail("Expand input shape is required.")
        expanded_shape = _effective_expand_shape(input_shape, target)
        if expanded_shape is None:
            expanded_shape = _shape(context.root.inputs[0])
        if expanded_shape is None or not all(isinstance(dim, int) for dim in expanded_shape):
            return result.fail("The effective Expand output shape must be static.")
        if len(input_shape) != len(expanded_shape):
            return result.fail("Expand must not prepend dimensions before Unsqueeze is moved.")
        normalized = _normalize_axes(inserted, len(expanded_shape) + len(inserted))
        if normalized is None:
            return result.fail("Unsqueeze axes are invalid.")
        self._target = list(expanded_shape)
        for axis in normalized:
            self._target.insert(axis, 1)
        return result

    def rewrite(self, op, x, expand_shape, axes):
        unsqueezed = op.Unsqueeze(x, axes)
        target = op.Constant(value=ir.tensor(self._target, dtype=ir.DataType.INT64))
        return op.Expand(unsqueezed, target)


class FuseExpandsAroundUnsqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("FuseExpandsAroundUnsqueeze")

    def pattern(self, op, x, shape1, axes, shape2):
        first = op.Expand(x, shape1)
        return op.Expand(op.Unsqueeze(first, axes), shape2)

    def check(self, context, x, shape1, axes, shape2):
        del context
        result = MatchResult()
        first = _ints(shape1)
        inserted = _ints(axes)
        second = _ints(shape2)
        if first is None or inserted is None or second is None:
            return result.fail("All shapes and axes must be constants.")
        input_shape = _shape(x)
        if input_shape is None:
            return result.fail("Expand input shape is required.")
        expanded_first = _effective_expand_shape(input_shape, first)
        if expanded_first is None:
            return result.fail("The first Expand output shape is not statically provable.")
        if len(input_shape) != len(expanded_first):
            return result.fail("The first Expand must not prepend dimensions.")
        normalized = _normalize_axes(inserted, len(expanded_first) + len(inserted))
        if normalized is None:
            return result.fail("Unsqueeze axes are invalid.")
        for axis in normalized:
            expanded_first.insert(axis, 1)
        combined = _effective_expand_shape(expanded_first, second)
        if combined is None:
            return result.fail("The second Expand target is incompatible.")
        self._target = combined
        return result

    def rewrite(self, op, x, shape1, axes, shape2):
        unsqueezed = op.Unsqueeze(x, axes)
        target = op.Constant(value=ir.tensor(self._target, dtype=ir.DataType.INT64))
        return op.Expand(unsqueezed, target)


class _MoveEqualExpandsAfterBinary(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"MoveEqualExpandsAfter{op_type}")
        self._op_type = op_type

    def pattern(self, op, x, shape1, y, shape2):
        return getattr(op, self._op_type)(op.Expand(x, shape1), op.Expand(y, shape2))

    def check(self, context, x, shape1, y, shape2):
        result = MatchResult()
        if shape1 is not shape2 and _ints(shape1) != _ints(shape2):
            return result.fail("Expand targets must be equal constants.")
        x_shape = _shape(x)
        y_shape = _shape(y)
        output_shape = _shape(context.output_values[0])
        if x_shape is None or y_shape is None or output_shape is None:
            return result.fail("Input and output shapes are required.")
        small_output = _broadcast(x_shape, y_shape)
        if small_output is None or small_output == output_shape:
            return result.fail("The smaller binary result must require one final Expand.")
        target = _ints(shape1)
        if target is None or target != output_shape:
            return result.fail("Constant target must equal the binary output shape.")
        if not _expand_preserves_nonempty_evaluation(small_output, output_shape):
            return result.fail(
                "The final Expand must not discard an otherwise evaluated binary result."
            )
        return result

    def rewrite(self, op, x, shape1, y, shape2):
        small = getattr(op, self._op_type)(x, y)
        return op.Expand(small, shape1)


class _MoveExpandAfterBinary(RewriteRuleClassBase):
    def __init__(self, op_type: str, expand_left: bool):
        side = "Left" if expand_left else "Right"
        super().__init__(f"Move{side}InputExpandAfter{op_type}")
        self._op_type = op_type
        self._expand_left = expand_left

    def pattern(self, op, x, expand_shape, y):
        if self._expand_left:
            return getattr(op, self._op_type)(op.Expand(x, expand_shape), y)
        return getattr(op, self._op_type)(x, op.Expand(y, expand_shape))

    def check(self, context, x: ir.Value, expand_shape: ir.Value, y: ir.Value):
        result = MatchResult()
        x_shape = _shape(x)
        y_shape = _shape(y)
        output_shape = _shape(context.output_values[0])
        target = _ints(expand_shape)
        if x_shape is None or y_shape is None or output_shape is None or target is None:
            return result.fail("Known shapes and a constant Expand target are required.")
        smaller = _broadcast(x_shape, y_shape)
        if smaller is None or smaller == output_shape or target != output_shape:
            return result.fail("The smaller binary result cannot be restored by one Expand.")
        if not _expand_preserves_nonempty_evaluation(smaller, output_shape):
            return result.fail(
                "The final Expand must not discard an otherwise evaluated binary result."
            )
        return result

    def rewrite(self, op, x, expand_shape, y):
        small = getattr(op, self._op_type)(x, y)
        return op.Expand(small, expand_shape)


class MoveReshapeBeforeExpand(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveReshapeBeforeExpand")

    def pattern(self, op, x, expand_shape, reshape_shape):
        return op.Reshape(op.Expand(x, expand_shape), reshape_shape)

    def check(self, context, x: ir.Value, expand_shape: ir.Value, reshape_shape: ir.Value):
        result = MatchResult()
        target = _ints(expand_shape)
        if (
            x.shape is None
            or x.shape.rank() != 3
            or target is None
            or len(target) != 3
            or target[1:] != [1, 1]
            or _ints(reshape_shape) != [0, 1, -1]
        ):
            return result.fail(
                "Expand/Reshape does not match the proven rank-three special case."
            )
        allowzero = context.root.attributes.get("allowzero")
        if allowzero is not None and allowzero.is_ref():
            return result.fail("allowzero is a reference attribute.")
        if allowzero is not None and allowzero.as_int() != 0:
            return result.fail("allowzero=1 is not supported.")
        return result

    def rewrite(self, op, x, expand_shape, reshape_shape):
        reshaped = op.Reshape(x, reshape_shape)
        return op.Expand(reshaped, expand_shape)


class _MoveWhereBeforeExpand(RewriteRuleClassBase):
    def __init__(self, expanded_branch: int):
        branch = "Then" if expanded_branch == 1 else "Else"
        super().__init__(f"MoveWhereBeforeExpand{branch}Branch")
        self._expanded_branch = expanded_branch

    def pattern(self, op, x, expand_shape, other, to):
        expanded = op.Expand(x, expand_shape)
        condition = op.Cast(expanded, to=to)
        if self._expanded_branch == 1:
            return op.Where(condition, expanded, other)
        return op.Where(condition, other, expanded)

    def check(
        self, context, x: ir.Value, expand_shape: ir.Value, other: ir.Value, to: ir.Attr
    ):
        result = MatchResult()
        if to.is_ref() or to.as_int() != int(ir.DataType.BOOL):
            return result.fail("Cast must produce a boolean Where condition.")
        x_shape = _shape(x)
        other_shape = _shape(other)
        output_shape = _shape(context.output_values[0])
        if x_shape is None or other_shape is None or output_shape is None:
            return result.fail("Input and output shapes are required.")
        small_shape = _broadcast(x_shape, other_shape)
        if small_shape is None or not _broadcasts_to(small_shape, output_shape):
            return result.fail("The smaller Where result does not broadcast to the output.")
        if not _expand_preserves_nonempty_evaluation(small_shape, output_shape):
            return result.fail(
                "The final Expand must not discard an otherwise evaluated Where result."
            )
        target = _ints(expand_shape)
        if target is not None and target != output_shape:
            return result.fail("Constant Expand target does not equal the Where output shape.")
        expanded = context.root.inputs[0].producer().inputs[0]
        if target is None:
            expanded_shape = _shape(expanded)
            if (
                expanded_shape is None
                or len(expanded_shape) != len(output_shape)
                or any(
                    not _ir_utils.same_dim(left, right)
                    for left, right in zip(expanded_shape, output_shape)
                )
            ):
                return result.fail(
                    "Expanded condition shape must equal the Where output shape."
                )
        cast = context.root.inputs[0].producer()
        self._cast_attributes = dict(cast.attributes)
        return result

    def rewrite(self, op, x, expand_shape, other, to):
        condition = op.Cast(x, **self._cast_attributes)
        if self._expanded_branch == 1:
            small = op.Where(condition, x, other)
        else:
            small = op.Where(condition, other, x)
        return op.Expand(small, expand_shape)


class _SimplifyConcatExpandShapeBase(RewriteRuleClassBase):
    def __init__(self, rank: int):
        super().__init__(f"SimplifyConcatExpandShapeRank{rank}", remove_nodes=False)
        self._rank = rank

    def _check(self, context, x: ir.Value, parts: tuple[ir.Value, ...], axis: ir.Attr):
        result = MatchResult()
        if axis.is_ref() or axis.as_int() != 0:
            return result.fail("Concat axis must be zero.")
        before = _shape(x)
        after = _shape(context.output_values[0])
        if (
            before is None
            or after is None
            or len(before) != self._rank
            or len(after) != self._rank
        ):
            return result.fail("Known equal input/output ranks are required.")
        changed = [i for i, (left, right) in enumerate(zip(before, after)) if left != right]
        if len(changed) != 1:
            return result.fail("Expand must change exactly one dimension.")
        for part in parts:
            if part.shape is None or list(part.shape) != [1]:
                return result.fail("Each Concat input must be a one-element vector.")
        self._changed = changed[0]
        return result

    def _rewrite(self, op, x, parts: tuple[ir.Value, ...]):
        one = op.Constant(value=ir.tensor([1], dtype=ir.DataType.INT64))
        target_parts = [one] * self._rank
        target_parts[self._changed] = parts[self._changed]
        target = op.Concat(*target_parts, axis=0)
        return op.Expand(x, target)


class SimplifyConcatExpandShapeRank2(_SimplifyConcatExpandShapeBase):
    def __init__(self):
        super().__init__(2)

    def pattern(self, op, x, p0, p1, axis):
        return op.Expand(x, op.Concat(p0, p1, axis=axis))

    def check(self, context, x, p0, p1, axis):
        return self._check(context, x, (p0, p1), axis)

    def rewrite(self, op, x, p0, p1, axis):
        return self._rewrite(op, x, (p0, p1))


class SimplifyConcatExpandShapeRank3(_SimplifyConcatExpandShapeBase):
    def __init__(self):
        super().__init__(3)

    def pattern(self, op, x, p0, p1, p2, axis):
        return op.Expand(x, op.Concat(p0, p1, p2, axis=axis))

    def check(self, context, x, p0, p1, p2, axis):
        return self._check(context, x, (p0, p1, p2), axis)

    def rewrite(self, op, x, p0, p1, p2, axis):
        return self._rewrite(op, x, (p0, p1, p2))


class SimplifyConcatExpandShapeRank4(_SimplifyConcatExpandShapeBase):
    def __init__(self):
        super().__init__(4)

    def pattern(self, op, x, p0, p1, p2, p3, axis):
        return op.Expand(x, op.Concat(p0, p1, p2, p3, axis=axis))

    def check(self, context, x, p0, p1, p2, p3, axis):
        return self._check(context, x, (p0, p1, p2, p3), axis)

    def rewrite(self, op, x, p0, p1, p2, p3, axis):
        return self._rewrite(op, x, (p0, p1, p2, p3))


class SimplifyConcatExpandShapeRank5(_SimplifyConcatExpandShapeBase):
    def __init__(self):
        super().__init__(5)

    def pattern(self, op, x, p0, p1, p2, p3, p4, axis):
        return op.Expand(x, op.Concat(p0, p1, p2, p3, p4, axis=axis))

    def check(self, context, x, p0, p1, p2, p3, p4, axis):
        return self._check(context, x, (p0, p1, p2, p3, p4), axis)

    def rewrite(self, op, x, p0, p1, p2, p3, p4, axis):
        return self._rewrite(op, x, (p0, p1, p2, p3, p4))


_SAFE_UNARY = (
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Cos",
    "Cosh",
    "Erf",
    "Exp",
    "Floor",
    "IsInf",
    "IsNaN",
    "Log",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
)

_SAFE_BINARY = (
    "Add",
    "And",
    "BitShift",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "Sub",
    "Xor",
)

rules = RewriteRuleSet(
    [
        MaterializeExpandShape.rule(),
        ReplaceUnitExpandWithUnsqueeze.rule(),
        _MoveWhereBeforeExpand.rule(1),
        _MoveWhereBeforeExpand.rule(2),
        SimplifyConcatExpandShapeRank2.rule(),
        SimplifyConcatExpandShapeRank3.rule(),
        SimplifyConcatExpandShapeRank4.rule(),
        SimplifyConcatExpandShapeRank5.rule(),
        *[_MoveUnaryBeforeExpand.rule(op_type) for op_type in _SAFE_UNARY],
        MoveCastBeforeExpand.rule(),
        MoveUnsqueezeBeforeExpand.rule(),
        FuseExpandsAroundUnsqueeze.rule(),
        *[_MoveEqualExpandsAfterBinary.rule(op_type) for op_type in _SAFE_BINARY],
        *[
            _MoveExpandAfterBinary.rule(op_type, side)
            for op_type in _SAFE_BINARY
            for side in (True, False)
        ],
        MoveReshapeBeforeExpand.rule(),
    ]
)
