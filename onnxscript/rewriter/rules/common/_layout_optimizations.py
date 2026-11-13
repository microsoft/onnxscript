# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Sound layout and shape-manipulation rewrite rules."""

from __future__ import annotations

import math
from collections.abc import Sequence

import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _ints(value: ir.Value) -> list[int] | None:
    array = _ir_utils.get_numpy_value(value)
    if array is None or array.ndim != 1 or array.dtype.kind not in "iu":
        return None
    return [int(v) for v in array.reshape(-1)]


def _valid_perm(perm: Sequence[int], rank: int | None = None) -> bool:
    return (rank is None or len(perm) == rank) and sorted(perm) == list(range(len(perm)))


def _normalize_axes(axes: Sequence[int], rank: int) -> list[int] | None:
    result = []
    for axis in axes:
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank or normalized in result:
            return None
        result.append(normalized)
    return sorted(result)


def _static_shape(value: ir.Value) -> list[int] | None:
    if value.shape is None or value.shape.rank() is None:
        return None
    dims = list(value.shape)
    return [int(d) for d in dims] if all(isinstance(d, int) for d in dims) else None


def _known_shape(value: ir.Value) -> list | None:
    if value.shape is None or value.shape.rank() is None:
        return None
    dims = list(value.shape)
    if any(isinstance(dim, ir.SymbolicDim) and dim.value is None for dim in dims):
        return None
    return dims


def _same_known_dims(left: Sequence, right: Sequence) -> bool:
    return len(left) == len(right) and all(
        _ir_utils.same_dim(left_dim, right_dim) for left_dim, right_dim in zip(left, right)
    )


def _int_attribute(node: ir.Node, name: str, default: int) -> int | None:
    attribute = node.attributes.get(name)
    if attribute is None:
        return default
    if attribute.is_ref():
        return None
    value = int(attribute.as_int())
    return value if value in (0, 1) else None


def _default_opset(context) -> int:
    imports = context.graph_or_function.opset_imports
    return imports.get(
        "",
        imports.get(
            "ai.onnx",
            context.model.opset_imports.get("", context.model.opset_imports.get("ai.onnx", 0)),
        ),
    )


def _resolve_reshape_shape(
    target: Sequence[int], input_shape: Sequence[int], allowzero: int
) -> list[int] | None:
    resolved = []
    inferred = None
    for index, dim in enumerate(target):
        if dim == 0 and not allowzero:
            if index >= len(input_shape):
                return None
            resolved.append(input_shape[index])
        elif dim == -1:
            if inferred is not None:
                return None
            inferred = index
            resolved.append(-1)
        elif dim >= 0:
            resolved.append(dim)
        else:
            return None
    if allowzero and inferred is not None and 0 in resolved:
        return None
    if inferred is not None:
        known = math.prod(dim for dim in resolved if dim != -1)
        total = math.prod(input_shape)
        if known == 0 or total % known:
            return None
        resolved[inferred] = total // known
    elif math.prod(resolved) != math.prod(input_shape):
        return None
    return resolved


def _reshape_inference_has_nonzero_denominator(
    target: Sequence[int], input_shape: Sequence, allowzero: int
) -> bool:
    if -1 not in target:
        return True
    for index, dim in enumerate(target):
        if dim == -1:
            continue
        if dim == 0 and allowzero == 0:
            if (
                index >= len(input_shape)
                or not isinstance(input_shape[index], int)
                or input_shape[index] <= 0
            ):
                return False
        elif dim <= 0:
            return False
    return True


class _EliminateSqueezeUnsqueezeAroundBinary(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"EliminateSqueezeUnsqueezeAround{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, squeeze_axes, other, unsqueeze_axes):
        squeezed = op.Squeeze(x, squeeze_axes)
        result = getattr(op, self._op_type)(squeezed, other)
        return op.Unsqueeze(result, unsqueeze_axes)

    def check(self, context, x, squeeze_axes, other, unsqueeze_axes):
        result = MatchResult()
        binary = context.root.inputs[0].producer()
        if binary is None or any(
            attribute.is_ref() for attribute in binary.attributes.values()
        ):
            return result.fail("Binary attributes must be concrete.")
        if _static_shape(x) != [1]:
            return result.fail("The squeezed input must have static shape [1].")
        if other.shape is None or other.shape.rank() != 0:
            return result.fail("The other operand must be scalar.")
        if _normalize_axes(_ints(squeeze_axes) or [], 1) != [0]:
            return result.fail("Squeeze must remove axis zero.")
        if _normalize_axes(_ints(unsqueeze_axes) or [], 1) != [0]:
            return result.fail("Unsqueeze must restore axis zero.")
        self._attributes = dict(binary.attributes)
        return result

    def rewrite(self, op, x, squeeze_axes, other, unsqueeze_axes):
        expanded_other = op.Unsqueeze(other, unsqueeze_axes)
        return getattr(op, self._op_type)(x, expanded_other, **self._attributes)


class MoveSqueezeAfterAdd(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveSqueezeAfterAdd", remove_nodes=False)

    def pattern(self, op, x, x_axes, y, y_axes):
        return op.Add(op.Squeeze(x, x_axes), op.Squeeze(y, y_axes))

    def check(self, context, x, x_axes, y, y_axes):
        del context
        result = MatchResult()
        x_shape = _known_shape(x)
        y_shape = _known_shape(y)
        if x_shape is None or y_shape is None or len(x_shape) != len(y_shape):
            return result.fail("Squeeze inputs must have equal known ranks.")
        x_normalized = _normalize_axes(_ints(x_axes) or [], len(x_shape))
        y_normalized = _normalize_axes(_ints(y_axes) or [], len(y_shape))
        if not x_normalized or x_normalized != y_normalized:
            return result.fail("Squeeze axes must normalize to the same dimensions.")
        if any(x_shape[axis] != 1 or y_shape[axis] != 1 for axis in x_normalized):
            return result.fail("Squeezed dimensions must be statically one.")
        return result

    def rewrite(self, op, x, x_axes, y, y_axes):
        return op.Squeeze(op.Add(x, y), x_axes)


class MoveUnsqueezeAfterMul(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveUnsqueezeAfterMul", remove_nodes=False)

    def pattern(self, op, x, x_axes, y, y_axes):
        return op.Mul(op.Unsqueeze(x, x_axes), op.Unsqueeze(y, y_axes))

    def check(self, context, x, x_axes, y, y_axes):
        del context
        result = MatchResult()
        x_shape = _known_shape(x)
        y_shape = _known_shape(y)
        if x_shape is None or y_shape is None or len(x_shape) != len(y_shape):
            return result.fail("Unsqueeze inputs must have equal known ranks.")
        x_values = _ints(x_axes)
        y_values = _ints(y_axes)
        if x_values is None or y_values is None:
            return result.fail("Unsqueeze axes must be constants.")
        x_normalized = _normalize_axes(x_values, len(x_shape) + len(x_values))
        y_normalized = _normalize_axes(y_values, len(y_shape) + len(y_values))
        if not x_normalized or x_normalized != y_normalized:
            return result.fail("Unsqueeze axes must normalize to the same dimensions.")
        return result

    def rewrite(self, op, x, x_axes, y, y_axes):
        return op.Unsqueeze(op.Mul(x, y), x_axes)


class MoveTransposeBeforeUnsqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveTransposeBeforeUnsqueeze", remove_nodes=False)

    def pattern(self, op, x, axes, perm):
        return op.Transpose(op.Unsqueeze(x, axes), perm=perm)

    def check(self, context, x: ir.Value, axes: ir.Value, perm: ir.Attr):
        del context
        result = MatchResult()
        if perm.is_ref():
            return result.fail("Permutation is a reference attribute.")
        permutation = list(perm.as_ints())
        if not _valid_perm(permutation):
            return result.fail("Permutation is invalid.")
        normalized = _normalize_axes(_ints(axes) or [], len(permutation))
        if not normalized:
            return result.fail("Unsqueeze axes are invalid.")
        if x.shape is not None and x.shape.rank() != len(permutation) - len(normalized):
            return result.fail("Input rank does not agree with the permutation.")
        self._new_axes = [i for i, p in enumerate(permutation) if p in normalized]
        old_to_new = {}
        next_axis = 0
        for i in range(len(permutation)):
            if i not in normalized:
                old_to_new[i] = next_axis
                next_axis += 1
        self._new_perm = [old_to_new[p] for p in permutation if p not in normalized]
        return result

    def rewrite(self, op, x, axes, perm):
        new_axes = op.Constant(value=ir.tensor(self._new_axes, dtype=ir.DataType.INT64))
        transposed = op.Transpose(x, perm=self._new_perm)
        return op.Unsqueeze(transposed, new_axes)


class MoveScalarGatherBeforeTranspose(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveScalarGatherBeforeTranspose", remove_nodes=False)

    def pattern(self, op, x, indices, perm, axis):
        return op.Gather(op.Transpose(x, perm=perm), indices, axis=axis)

    def check(self, context, x, indices: ir.Value, perm: ir.Attr, axis: ir.Attr):
        del context, x
        result = MatchResult()
        if indices.shape is None or indices.shape.rank() != 0:
            return result.fail("Gather indices must be scalar.")
        if perm.is_ref() or axis.is_ref():
            return result.fail("Attributes must be concrete.")
        permutation = list(perm.as_ints())
        if not _valid_perm(permutation):
            return result.fail("Permutation is invalid.")
        old_axis = int(axis.as_int())
        if old_axis < 0:
            old_axis += len(permutation)
        if old_axis < 0 or old_axis >= len(permutation):
            return result.fail("Gather axis is out of range.")
        self._new_axis = permutation[old_axis]
        remaining = [p for i, p in enumerate(permutation) if i != old_axis]
        self._new_perm = [p - (p > self._new_axis) for p in remaining]
        return result

    def rewrite(self, op, x, indices, perm, axis):
        gathered = op.Gather(x, indices, axis=self._new_axis)
        if self._new_perm == list(range(len(self._new_perm))):
            return gathered
        return op.Transpose(gathered, perm=self._new_perm)


def _check_transpose_gather(context, indices: ir.Value, rule) -> MatchResult:
    result = MatchResult()
    if indices.shape is None or indices.shape.rank() != 0:
        return result.fail("Gather indices must be scalar.")
    gather = context.root
    transpose = gather.inputs[0].producer()
    if transpose is None or transpose.op_type != "Transpose":
        return result.fail("Gather input must be produced by Transpose.")
    rank = transpose.inputs[0].shape.rank() if transpose.inputs[0].shape is not None else None
    perm_attr = transpose.attributes.get("perm")
    if perm_attr is not None and perm_attr.is_ref():
        return result.fail("Permutation is a reference attribute.")
    if perm_attr is None:
        if rank is None:
            return result.fail("Input rank is required for the default Transpose permutation.")
        permutation = list(reversed(range(rank)))
    else:
        permutation = list(perm_attr.as_ints())
    if not _valid_perm(permutation, rank):
        return result.fail("Permutation is invalid.")
    axis_attr = gather.attributes.get("axis")
    if axis_attr is not None and axis_attr.is_ref():
        return result.fail("Gather axis is a reference attribute.")
    old_axis = 0 if axis_attr is None else int(axis_attr.as_int())
    if old_axis < 0:
        old_axis += len(permutation)
    if old_axis < 0 or old_axis >= len(permutation):
        return result.fail("Gather axis is out of range.")
    rule._new_axis = permutation[old_axis]
    remaining = [p for i, p in enumerate(permutation) if i != old_axis]
    rule._new_perm = [p - (p > rule._new_axis) for p in remaining]
    return result


class MoveScalarGatherBeforeTransposeDefaultAxis(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("MoveScalarGatherBeforeTransposeDefaultAxis", remove_nodes=False)

    def pattern(self, op, x, indices, perm):
        return op.Gather(op.Transpose(x, perm=perm), indices)

    def check(self, context, x, indices, perm):
        del x, perm
        return _check_transpose_gather(context, indices, self)

    def rewrite(self, op, x, indices, perm):
        gathered = op.Gather(x, indices, axis=self._new_axis)
        if self._new_perm == list(range(len(self._new_perm))):
            return gathered
        return op.Transpose(gathered, perm=self._new_perm)


class MoveScalarGatherBeforeTransposeDefaultPermutation(RewriteRuleClassBase):
    def __init__(self):
        super().__init__(
            "MoveScalarGatherBeforeTransposeDefaultPermutation", remove_nodes=False
        )

    def pattern(self, op, x, indices, axis):
        return op.Gather(op.Transpose(x), indices, axis=axis)

    def check(self, context, x, indices, axis):
        del x, axis
        return _check_transpose_gather(context, indices, self)

    def rewrite(self, op, x, indices, axis):
        gathered = op.Gather(x, indices, axis=self._new_axis)
        if self._new_perm == list(range(len(self._new_perm))):
            return gathered
        return op.Transpose(gathered, perm=self._new_perm)


class MoveScalarGatherBeforeTransposeDefaultAxisAndPermutation(RewriteRuleClassBase):
    def __init__(self):
        super().__init__(
            "MoveScalarGatherBeforeTransposeDefaultAxisAndPermutation",
            remove_nodes=False,
        )

    def pattern(self, op, x, indices):
        return op.Gather(op.Transpose(x), indices)

    def check(self, context, x, indices):
        del x
        return _check_transpose_gather(context, indices, self)

    def rewrite(self, op, x, indices):
        gathered = op.Gather(x, indices, axis=self._new_axis)
        if self._new_perm == list(range(len(self._new_perm))):
            return gathered
        return op.Transpose(gathered, perm=self._new_perm)


class ReplaceUnitDimensionTransposeWithReshape(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("ReplaceUnitDimensionTransposeWithReshape")

    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, context, x: ir.Value, perm: ir.Attr):
        result = MatchResult()
        shape = _static_shape(x)
        if shape is None or perm.is_ref():
            return result.fail("A static input shape and concrete permutation are required.")
        permutation = list(perm.as_ints())
        if not _valid_perm(permutation, len(shape)):
            return result.fail("Permutation is invalid.")
        moved_non_unit = {
            axis
            for output_axis, axis in enumerate(permutation)
            if output_axis != axis and shape[axis] != 1
        }
        if len(moved_non_unit) > 1:
            return result.fail("Transpose reorders more than one non-unit dimension.")
        self._shape = [shape[p] for p in permutation]
        self._allowzero = 1 if 0 in self._shape else None
        if self._allowzero and _default_opset(context) < 14:
            return result.fail("Literal zero Reshape dimensions require opset 14 or later.")
        return result

    def rewrite(self, op, x, perm):
        shape = op.Constant(value=ir.tensor(self._shape, dtype=ir.DataType.INT64))
        return op.Reshape(x, shape, allowzero=self._allowzero)


class FuseConsecutiveUnsqueezes(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("FuseConsecutiveUnsqueezes", remove_nodes=False)

    def pattern(self, op, x, axes1, axes2):
        return op.Unsqueeze(op.Unsqueeze(x, axes1), axes2)

    def check(self, context, x: ir.Value, axes1: ir.Value, axes2: ir.Value):
        del context
        result = MatchResult()
        first = _ints(axes1)
        second = _ints(axes2)
        if first is None or second is None:
            return result.fail("Axes must be constants.")
        input_rank = x.shape.rank() if x.shape is not None else None
        if input_rank is None:
            return result.fail("Input rank is required.")
        first = _normalize_axes(first, input_rank + len(first))
        if first is None:
            return result.fail("First axes are invalid.")
        second = _normalize_axes(second, input_rank + len(first) + len(second))
        if second is None:
            return result.fail("Second axes are invalid.")
        markers = [False] * input_rank
        for axis in first:
            markers.insert(axis, True)
        for axis in second:
            markers.insert(axis, True)
        self._axes = [i for i, inserted in enumerate(markers) if inserted]
        return result

    def rewrite(self, op, x, axes1, axes2):
        axes = op.Constant(value=ir.tensor(self._axes, dtype=ir.DataType.INT64))
        return op.Unsqueeze(x, axes)


class FuseConsecutiveUnsqueezeSqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("FuseConsecutiveUnsqueezeSqueeze", remove_nodes=False)

    def pattern(self, op, x, unsqueeze_axes, squeeze_axes):
        return op.Squeeze(op.Unsqueeze(x, unsqueeze_axes), squeeze_axes)

    def check(self, context, x: ir.Value, unsqueeze_axes: ir.Value, squeeze_axes: ir.Value):
        del context
        result = MatchResult()
        first = _ints(unsqueeze_axes)
        second = _ints(squeeze_axes)
        input_rank = x.shape.rank() if x.shape is not None else None
        if not first or not second or input_rank is None:
            return result.fail("Non-empty constant axes and input rank are required.")
        output_rank = input_rank + len(first)
        first = _normalize_axes(first, output_rank)
        second = _normalize_axes(second, output_rank)
        if first is None or second is None or not set(first).issubset(second):
            return result.fail("Squeeze must remove all inserted axes.")
        remaining = [axis for axis in second if axis not in first]
        self._axes = [axis - sum(inserted < axis for inserted in first) for axis in remaining]
        if self._axes:
            shape = _static_shape(x)
            if shape is None or any(shape[axis] != 1 for axis in self._axes):
                return result.fail("Remaining squeeze axes are not provably unit dimensions.")
        return result

    def rewrite(self, op, x, unsqueeze_axes, squeeze_axes):
        if not self._axes:
            return op.Identity(x)
        axes = op.Constant(value=ir.tensor(self._axes, dtype=ir.DataType.INT64))
        return op.Squeeze(x, axes)


class FuseReshapeSqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("FuseReshapeSqueeze", remove_nodes=False)

    def pattern(self, op, x, shape, axes):
        return op.Squeeze(op.Reshape(x, shape), axes)

    def check(self, context, x, shape: ir.Value, axes: ir.Value):
        result = MatchResult()
        target = _ints(shape)
        squeeze_axes = _ints(axes)
        reshape = context.root.inputs[0].producer()
        if reshape is None or reshape.op_type != "Reshape":
            return result.fail("Matched input is not produced by Reshape.")
        if target is None or squeeze_axes is None:
            return result.fail("Shape and axes must be constants.")
        allowzero = _int_attribute(reshape, "allowzero", 0)
        if allowzero is None:
            return result.fail("allowzero is a reference attribute.")
        if allowzero != 0:
            return result.fail("allowzero=1 is not supported.")
        normalized = _normalize_axes(squeeze_axes, len(target))
        if not normalized or any(target[axis] != 1 for axis in normalized):
            return result.fail(
                "Squeeze axes must be non-empty and select literal unit dimensions."
            )
        if any(
            target[i] == 0 for i in range(min(normalized), len(target)) if i not in normalized
        ):
            return result.fail("A copied dimension follows a removed dimension.")
        self._shape = [d for i, d in enumerate(target) if i not in normalized]
        return result

    def rewrite(self, op, x, shape, axes):
        new_shape = op.Constant(value=ir.tensor(self._shape, dtype=ir.DataType.INT64))
        return op.Reshape(x, new_shape)


class RemoveIdentityReshape(RewriteRuleClassBase):
    """Remove a Reshape whose inferred output shape equals its input shape."""

    def __init__(self):
        super().__init__("RemoveIdentityReshape")

    def pattern(self, op, x, shape):
        return op.Reshape(x, shape)

    def check(self, context, x: ir.Value, shape: ir.Value):
        result = MatchResult()
        target = _ints(shape)
        input_shape = _known_shape(x)
        output_shape = _known_shape(context.output_values[0])
        if target is None or input_shape is None or output_shape is None:
            return result.fail(
                "Constant target and inferred input/output shapes are required."
            )
        if not _same_known_dims(input_shape, output_shape):
            return result.fail("Reshape changes the inferred shape.")
        allowzero = _int_attribute(context.root, "allowzero", 0)
        if allowzero is None:
            return result.fail("allowzero must be a concrete zero or one.")
        if any(dim < -1 for dim in target) or target.count(-1) > 1:
            return result.fail("Reshape target is invalid.")
        if allowzero and 0 in target and -1 in target:
            return result.fail("allowzero=1 cannot combine zero and inferred dimensions.")
        if len(target) != len(input_shape):
            return result.fail("Identity Reshape must preserve rank.")
        inferred_axis = None
        for i, (requested, input_dim) in enumerate(zip(target, input_shape)):
            if requested == -1:
                inferred_axis = i
            elif requested == 0 and not allowzero:
                continue
            elif not _ir_utils.same_dim(requested, input_dim):
                return result.fail("Reshape target does not reproduce the input shape.")
        if inferred_axis is not None:
            if not _reshape_inference_has_nonzero_denominator(target, input_shape, allowzero):
                return result.fail("The inferred dimension may have a zero denominator.")
        return result

    def rewrite(self, op, x, shape):
        return op.Identity(x)


class ReplaceUnitDimensionReshapeWithSqueezeOrUnsqueeze(RewriteRuleClassBase):
    """Replace Reshape with Squeeze/Unsqueeze when only unit axes change."""

    def __init__(self):
        super().__init__("ReplaceUnitDimensionReshapeWithSqueezeOrUnsqueeze")

    def pattern(self, op, x, shape):
        return op.Reshape(x, shape)

    def check(self, context, x: ir.Value, shape: ir.Value):
        result = MatchResult()
        opset = _default_opset(context)
        if opset < 13:
            return result.fail("Axes-as-input Squeeze/Unsqueeze requires opset 13 or later.")
        target = _ints(shape)
        allowzero = _int_attribute(context.root, "allowzero", 0)
        if target is None or allowzero is None:
            return result.fail("A constant valid Reshape target is required.")
        if (
            any(dim < -1 for dim in target)
            or target.count(-1) > 1
            or (allowzero == 1 and 0 in target and -1 in target)
        ):
            return result.fail("Reshape target is invalid.")
        before = _shape_dims(x)
        after = _shape_dims(context.output_values[0])
        if before is None or after is None or len(before) == len(after):
            return result.fail("Known shapes with different ranks are required.")
        if len(target) != len(after):
            return result.fail("Reshape target rank does not match the inferred output.")
        if not _reshape_inference_has_nonzero_denominator(target, before, allowzero):
            return result.fail("The inferred dimension may have a zero denominator.")
        for index, (requested, inferred) in enumerate(zip(target, after)):
            if requested == -1:
                continue
            if requested == 0 and allowzero == 0:
                if index >= len(before) or not _ir_utils.same_dim(before[index], inferred):
                    return result.fail("A copied Reshape dimension is not preserved.")
            elif not _ir_utils.same_dim(requested, inferred):
                return result.fail("Reshape target disagrees with the inferred output.")
        if len(before) < len(after):
            axes = _inserted_unit_axes(before, after)
            self._op_type = "Unsqueeze"
        else:
            axes = _inserted_unit_axes(after, before)
            self._op_type = "Squeeze"
        if axes is None:
            return result.fail("Shapes do not differ only by unit dimensions.")
        self._axes = axes
        return result

    def rewrite(self, op, x, shape):
        axes = op.Constant(value=ir.tensor(self._axes, dtype=ir.DataType.INT64))
        return getattr(op, self._op_type)(x, axes)


class RemoveInverseSqueezeUnsqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("RemoveInverseSqueezeUnsqueeze", remove_nodes=False)

    def pattern(self, op, x, squeeze_axes, unsqueeze_axes):
        return op.Unsqueeze(op.Squeeze(x, squeeze_axes), unsqueeze_axes)

    def check(self, context, x: ir.Value, squeeze_axes: ir.Value, unsqueeze_axes: ir.Value):
        result = MatchResult()
        shape = _known_shape(x)
        squeeze = _ints(squeeze_axes)
        unsqueeze = _ints(unsqueeze_axes)
        output_shape = _known_shape(context.output_values[0])
        if shape is None or squeeze is None or unsqueeze is None or output_shape is None:
            return result.fail("Known shapes and constant axes are required.")
        squeeze = _normalize_axes(squeeze, len(shape))
        if not squeeze or any(shape[axis] != 1 for axis in squeeze):
            return result.fail(
                "Squeeze axes must be non-empty and select statically unit dimensions."
            )
        squeezed_rank = len(shape) - len(squeeze)
        unsqueeze = _normalize_axes(unsqueeze, squeezed_rank + len(unsqueeze))
        if not unsqueeze:
            return result.fail("Unsqueeze axes must be non-empty and valid.")
        if not _same_known_dims(shape, output_shape):
            return result.fail("The pair does not restore the original shape.")
        return result

    def rewrite(self, op, x, squeeze_axes, unsqueeze_axes):
        return op.Identity(x)


class RemoveShapeOpBeforeReshape(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"Remove{op_type}BeforeReshape", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, axes, shape):
        changed = getattr(op, self._op_type)(x, axes)
        return op.Reshape(changed, shape)

    def check(self, context, x, axes, shape):
        del x, axes
        result = MatchResult()
        target = _ints(shape)
        if target is None or 0 in target:
            return result.fail("A constant target without copied zero dimensions is required.")
        allowzero = _int_attribute(context.root, "allowzero", 0)
        if allowzero is None:
            return result.fail("allowzero must be a concrete zero or one.")
        if allowzero != 0:
            return result.fail("allowzero=1 is not supported.")
        return result

    def rewrite(self, op, x, axes, shape):
        return op.Reshape(x, shape)


class ReplaceUnsqueezeReshapeWithUnsqueeze(RewriteRuleClassBase):
    def __init__(self):
        super().__init__("ReplaceUnsqueezeReshapeWithUnsqueeze", remove_nodes=False)

    def pattern(self, op, x, axes, shape):
        return op.Reshape(op.Unsqueeze(x, axes), shape)

    def check(self, context, x: ir.Value, axes: ir.Value, shape: ir.Value):
        result = MatchResult()
        if x.shape is None or x.shape.rank() != 3:
            return result.fail("Source rank must be three.")
        if _ints(axes) != [2] or _ints(shape) != [0, 1, -1, 0]:
            return result.fail("Axes and target do not match the supported safe case.")
        allowzero = _int_attribute(context.root, "allowzero", 0)
        if allowzero is None:
            return result.fail("allowzero is a reference attribute.")
        if allowzero != 0:
            return result.fail("allowzero=1 is not supported.")
        source_shape = _known_shape(x)
        if source_shape is None or any(
            not isinstance(source_shape[axis], int) or source_shape[axis] <= 0
            for axis in (0, 2)
        ):
            return result.fail("Copied dimensions around -1 must be statically positive.")
        return result

    def rewrite(self, op, x, axes, shape):
        new_axes = op.Constant(value=ir.tensor([1], dtype=ir.DataType.INT64))
        return op.Unsqueeze(x, new_axes)


class _MoveEqualReshapesAfterBinary(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"MoveEqualReshapesAfter{op_type}", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, shape1, y, shape2):
        return getattr(op, self._op_type)(op.Reshape(x, shape1), op.Reshape(y, shape2))

    def check(self, context, x: ir.Value, shape1: ir.Value, y: ir.Value, shape2: ir.Value):
        result = MatchResult()
        if any(attribute.is_ref() for attribute in context.root.attributes.values()):
            return result.fail("Binary attributes must be concrete.")
        if _ints(shape1) is None or _ints(shape1) != _ints(shape2):
            return result.fail("Reshape targets must be equal constants.")
        x_shape = _known_shape(x)
        y_shape = _known_shape(y)
        if x_shape is None or y_shape is None or not _same_known_dims(x_shape, y_shape):
            return result.fail("Source shapes must be equal.")
        left_reshape = context.root.inputs[0].producer()
        right_reshape = context.root.inputs[1].producer()
        if left_reshape is None or right_reshape is None:
            return result.fail("Binary inputs must be produced by Reshape.")
        left_allowzero = _int_attribute(left_reshape, "allowzero", 0)
        right_allowzero = _int_attribute(right_reshape, "allowzero", 0)
        if left_allowzero is None or right_allowzero is None:
            return result.fail("allowzero must not be a reference attribute.")
        if left_allowzero != right_allowzero:
            return result.fail("Reshape allowzero settings must be equal.")
        self._allowzero = left_allowzero or None
        self._attributes = dict(context.root.attributes)
        return result

    def rewrite(self, op, x, shape1, y, shape2):
        result = getattr(op, self._op_type)(x, y, **self._attributes)
        return op.Reshape(result, shape1, allowzero=self._allowzero)


class _RemoveIdentityReshapesAroundBinary(RewriteRuleClassBase):
    def __init__(self, op_type: str, reshape_left: bool):
        side = "Left" if reshape_left else "Right"
        super().__init__(f"Remove{side}IdentityReshapesAround{op_type}", remove_nodes=False)
        self._op_type = op_type
        self._reshape_left = reshape_left

    def pattern(self, op, x, inner_shape, y, outer_shape):
        if self._reshape_left:
            result = getattr(op, self._op_type)(op.Reshape(x, inner_shape), y)
        else:
            result = getattr(op, self._op_type)(x, op.Reshape(y, inner_shape))
        return op.Reshape(result, outer_shape)

    def check(self, context, x: ir.Value, inner_shape, y: ir.Value, outer_shape):
        del inner_shape, outer_shape
        result = MatchResult()
        x_shape = _known_shape(x)
        y_shape = _known_shape(y)
        output_shape = _known_shape(context.output_values[0])
        if (
            x_shape is None
            or y_shape is None
            or output_shape is None
            or not _same_known_dims(x_shape, y_shape)
            or not _same_known_dims(output_shape, x_shape)
        ):
            return result.fail("Both source shapes and the final output shape must be equal.")
        binary = context.root.inputs[0].producer()
        binary_shape = _known_shape(binary.outputs[0]) if binary is not None else None
        if binary is None or binary_shape is None:
            return result.fail("The original binary output shape must be known.")
        if any(
            (shape := _known_shape(value)) is None or not _same_known_dims(shape, binary_shape)
            for value in binary.inputs
        ):
            return result.fail("The original binary operation must not broadcast.")
        if any(attribute.is_ref() for attribute in binary.attributes.values()):
            return result.fail("Binary attributes must be concrete.")
        self._attributes = dict(binary.attributes)
        return result

    def rewrite(self, op, x, inner_shape, y, outer_shape):
        return getattr(op, self._op_type)(x, y, **self._attributes)


class _ReduceReshape(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"Fuse{op_type}Reshape", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, axes, shape, keepdims):
        reduced = getattr(op, self._op_type)(x, axes, keepdims=keepdims)
        return op.Reshape(reduced, shape)

    def check(self, context, x: ir.Value, axes: ir.Value, shape, keepdims: ir.Attr):
        del shape
        result = MatchResult()
        if keepdims.is_ref() or keepdims.as_int() != 1:
            return result.fail("Reduction must keep dimensions.")
        input_shape = _known_shape(x)
        if input_shape is None:
            return result.fail("Reduction input rank is required.")
        normalized = _normalize_axes(_ints(axes) or [], len(input_shape))
        if not normalized:
            return result.fail("Reduction axes must be non-empty constants.")
        expected = [dim for i, dim in enumerate(input_shape) if i not in normalized]
        output_shape = _known_shape(context.output_values[0])
        if output_shape is None or not _same_known_dims(output_shape, expected):
            return result.fail(
                "Final shape does not equal the input with reduced axes removed."
            )
        return result

    def rewrite(self, op, x, axes, shape, keepdims):
        return getattr(op, self._op_type)(x, axes, keepdims=0)


class _ReduceReshapeAttributeAxes(RewriteRuleClassBase):
    def __init__(self, op_type: str):
        super().__init__(f"Fuse{op_type}ReshapeAttributeAxes", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, shape, axes, keepdims):
        reduced = getattr(op, self._op_type)(x, axes=axes, keepdims=keepdims)
        return op.Reshape(reduced, shape)

    def check(self, context, x: ir.Value, shape, axes: ir.Attr, keepdims: ir.Attr):
        del shape
        result = MatchResult()
        if axes.is_ref() or keepdims.is_ref() or keepdims.as_int() != 1:
            return result.fail(
                "Reduction attributes must be concrete and keepdims must be one."
            )
        input_shape = _known_shape(x)
        if input_shape is None:
            return result.fail("Reduction input rank is required.")
        normalized = _normalize_axes(list(axes.as_ints()), len(input_shape))
        if not normalized:
            return result.fail("Reduction axes must be non-empty constants.")
        expected = [dim for i, dim in enumerate(input_shape) if i not in normalized]
        output_shape = _known_shape(context.output_values[0])
        if output_shape is None or not _same_known_dims(output_shape, expected):
            return result.fail(
                "Final shape does not equal the input with reduced axes removed."
            )
        return result

    def rewrite(self, op, x, shape, axes, keepdims):
        return getattr(op, self._op_type)(x, axes=list(axes.as_ints()), keepdims=0)


class RemoveTransposeBeforeUnitReshapeTranspose(RewriteRuleClassBase):
    """Remove the leading Transpose when Reshape only edits unit dimensions."""

    def __init__(self):
        super().__init__("RemoveTransposeBeforeUnitReshapeTranspose", remove_nodes=False)

    def pattern(self, op, x, shape, perm1, perm2):
        first = op.Transpose(x, perm=perm1)
        reshaped = op.Reshape(first, shape)
        return op.Transpose(reshaped, perm=perm2)

    def check(self, context, x: ir.Value, shape: ir.Value, perm1: ir.Attr, perm2: ir.Attr):
        result = MatchResult()
        if perm1.is_ref() or perm2.is_ref() or _ints(shape) is None:
            return result.fail("Concrete permutations and Reshape target are required.")
        input_shape = _static_shape(x)
        reshape = context.root.inputs[0].producer()
        first = reshape.inputs[0].producer() if reshape is not None else None
        output_shape = _static_shape(context.output_values[0])
        if input_shape is None or output_shape is None or reshape is None:
            return result.fail("Static input and output shapes are required.")
        first_perm = list(perm1.as_ints())
        second_perm = list(perm2.as_ints())
        if not _valid_perm(first_perm, len(input_shape)):
            return result.fail("First Transpose permutation is invalid.")
        transposed_shape = [input_shape[axis] for axis in first_perm]
        allowzero = _int_attribute(reshape, "allowzero", 0)
        if allowzero is None:
            return result.fail("allowzero is a reference attribute.")
        reshaped_shape = _resolve_reshape_shape(
            _ints(shape) or [], transposed_shape, allowzero
        )
        if (
            first is None
            or reshaped_shape is None
            or not _valid_perm(second_perm, len(reshaped_shape))
        ):
            return result.fail("Transpose permutations are invalid.")

        non_unit_input = [(axis, dim) for axis, dim in enumerate(input_shape) if dim != 1]
        non_unit_transposed = [
            (first_perm[axis], dim) for axis, dim in enumerate(transposed_shape) if dim != 1
        ]
        if [dim for _, dim in non_unit_transposed] != [
            dim for dim in reshaped_shape if dim != 1
        ]:
            return result.fail("Reshape changes more than unit dimensions.")

        reshaped_tokens: list[int | None] = []
        token_index = 0
        for dim in reshaped_shape:
            if dim == 1:
                reshaped_tokens.append(None)
            else:
                reshaped_tokens.append(non_unit_transposed[token_index][0])
                token_index += 1
        desired_tokens = [reshaped_tokens[axis] for axis in second_perm]

        base_tokens: list[int | None] = [axis for axis, _ in non_unit_input]
        base_tokens.extend([None] * (len(output_shape) - len(base_tokens)))
        if len(base_tokens) != len(output_shape):
            return result.fail("Output rank cannot represent the non-unit input dimensions.")
        unused_units = [i for i, token in enumerate(base_tokens) if token is None]
        new_perm = []
        for token in desired_tokens:
            if token is None:
                if not unused_units:
                    return result.fail("Unit-dimension mapping is inconsistent.")
                new_perm.append(unused_units.pop(0))
            else:
                if token not in base_tokens:
                    return result.fail("Data-axis mapping is inconsistent.")
                new_perm.append(base_tokens.index(token))
        if not _valid_perm(new_perm, len(output_shape)):
            return result.fail("Derived permutation is invalid.")

        self._base_shape = [dim for _, dim in non_unit_input] + [1] * (
            len(output_shape) - len(non_unit_input)
        )
        self._allowzero = 1 if 0 in self._base_shape else None
        if self._allowzero and _default_opset(context) < 14:
            return result.fail("Literal zero Reshape dimensions require opset 14 or later.")
        self._new_perm = new_perm
        return result

    def rewrite(self, op, x, shape, perm1, perm2):
        new_shape = op.Constant(value=ir.tensor(self._base_shape, dtype=ir.DataType.INT64))
        reshaped = op.Reshape(x, new_shape, allowzero=self._allowzero)
        if self._new_perm == list(range(len(self._new_perm))):
            return reshaped
        return op.Transpose(reshaped, perm=self._new_perm)


def _shape_dims(value: ir.Value) -> list | None:
    return _known_shape(value)


def _inserted_unit_axes(smaller: list, larger: list) -> list[int] | None:
    """Return axes whose removal from larger yields smaller."""
    axes = []
    small_index = 0
    for large_index, dim in enumerate(larger):
        if small_index < len(smaller) and dim == smaller[small_index]:
            small_index += 1
        elif dim == 1:
            axes.append(large_index)
        else:
            return None
    return axes if small_index == len(smaller) else None


_squeeze_add_rule = MoveSqueezeAfterAdd.rule()
_unsqueeze_mul_rule = MoveUnsqueezeAfterMul.rule()
_swap_unsqueeze_transpose_rule = MoveTransposeBeforeUnsqueeze.rule()
_transpose_gather_rule = MoveScalarGatherBeforeTranspose.rule()
_transpose_gather_default_rules = [
    MoveScalarGatherBeforeTransposeDefaultAxis.rule(),
    MoveScalarGatherBeforeTransposeDefaultPermutation.rule(),
    MoveScalarGatherBeforeTransposeDefaultAxisAndPermutation.rule(),
]
_transpose_equal_reshape_rule = ReplaceUnitDimensionTransposeWithReshape.rule()
_unsqueeze_unsqueeze_rule = FuseConsecutiveUnsqueezes.rule()
_unsqueeze_squeeze_rule = FuseConsecutiveUnsqueezeSqueeze.rule()
_reshape_squeeze_rule = FuseReshapeSqueeze.rule()
_no_op_reshape_rule = RemoveIdentityReshape.rule()
_unit_dimension_reshape_rule = ReplaceUnitDimensionReshapeWithSqueezeOrUnsqueeze.rule()
_squeeze_unsqueeze_identity_rule = RemoveInverseSqueezeUnsqueeze.rule()
_remove_shape_before_reshape_rules = [
    RemoveShapeOpBeforeReshape.rule(op_type) for op_type in ("Squeeze", "Unsqueeze")
]
_unsqueeze_reshape_special_rule = ReplaceUnsqueezeReshapeWithUnsqueeze.rule()
_binary_ops = (
    "Add",
    "And",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Div",
    "Max",
    "Mean",
    "Min",
    "Mul",
    "Or",
    "Sub",
    "Sum",
    "Xor",
)
_equal_reshapes_binary_rules = [
    _MoveEqualReshapesAfterBinary.rule(op_type) for op_type in _binary_ops
]
_reshape_binary_reshape_rules = [
    _RemoveIdentityReshapesAroundBinary.rule(op_type, side)
    for op_type in _binary_ops
    for side in (True, False)
]
_reduce_reshape_rules = [
    _ReduceReshape.rule(op_type)
    for op_type in (
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    )
]
_reduce_reshape_attribute_rules = [
    _ReduceReshapeAttributeAxes.rule(op_type)
    for op_type in (
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    )
]
_transpose_unit_reshape_transpose_rule = RemoveTransposeBeforeUnitReshapeTranspose.rule()
_squeeze_binary_rules = [
    _EliminateSqueezeUnsqueezeAroundBinary.rule(op_type)
    for op_type in ("Add", "Div", "Mul", "Sub")
]

rules = RewriteRuleSet(
    [
        _squeeze_add_rule,
        _unsqueeze_mul_rule,
        *_squeeze_binary_rules,
        _swap_unsqueeze_transpose_rule,
        _transpose_gather_rule,
        *_transpose_gather_default_rules,
        _transpose_equal_reshape_rule,
        _unsqueeze_unsqueeze_rule,
        _unsqueeze_squeeze_rule,
        _squeeze_unsqueeze_identity_rule,
        _reshape_squeeze_rule,
        _unsqueeze_reshape_special_rule,
        *_remove_shape_before_reshape_rules,
        *_equal_reshapes_binary_rules,
        *_reshape_binary_reshape_rules,
        *_reduce_reshape_rules,
        *_reduce_reshape_attribute_rules,
        _transpose_unit_reshape_transpose_rule,
        _unit_dimension_reshape_rule,
        _no_op_reshape_rule,
    ]
)
