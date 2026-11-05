# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Compute shapes without processing tensor data."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import get_numpy_value, same_dim, same_shape
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _shape_attributes(node: ir.Node) -> dict[str, ir.Attr] | None:
    if any(attr.is_ref() for attr in node.attributes.values()):
        return None
    return dict(node.attributes)


def _shape_indices(node: ir.Node, rank: int) -> list[int]:
    return list(range(rank))[
        node.attributes.get_int("start", 0) : node.attributes.get_int("end", rank)
    ]


class ShapeCast(RewriteRuleClassBase):
    def pattern(self, op, x, to):
        return op.Shape(op.Cast(x, to=to))

    def check(self, context, x, to) -> MatchResult:
        self._attributes = _shape_attributes(context.root)
        result = MatchResult()
        if self._attributes is None:
            return result.fail("Shape attributes refer to function parameters.")
        return result

    def rewrite(self, op, x, to):
        return op.Shape(x, **self._attributes)


class ShapeTranspose(RewriteRuleClassBase):
    def pattern(self, op, x):
        return op.Shape(op.Transpose(x))

    def check(self, context, x) -> MatchResult:
        result = MatchResult()
        transpose = context.root.inputs[0].producer()
        if _shape_attributes(context.root) is None:
            return result.fail("Shape attributes refer to function parameters.")
        if any(attr.is_ref() for attr in transpose.attributes.values()):
            return result.fail("Transpose permutation is a reference attribute.")
        perm_attr = transpose.attributes.get("perm")
        if perm_attr is None:
            if x.shape is None:
                return result.fail("Input rank is needed for the default permutation.")
            perm = list(reversed(range(len(x.shape))))
        else:
            perm = list(perm_attr.as_ints())
        rank = len(perm)
        if sorted(perm) != list(range(rank)):
            return result.fail("Transpose permutation is invalid.")
        if x.shape is not None and len(x.shape) != rank:
            return result.fail("Transpose permutation disagrees with the input rank.")
        self._indices = [perm[i] for i in _shape_indices(context.root, rank)]
        return result

    def rewrite(self, op, x):
        indices = op.Constant(value=ir.tensor(self._indices, dtype=ir.DataType.INT64))
        if not self._indices:
            return indices
        return op.Gather(op.Shape(x), indices, axis=0)


class GatherShape(RewriteRuleClassBase):
    def pattern(self, op, x, indices):
        return op.Gather(op.Shape(x), indices)

    def check(self, context, x, indices) -> MatchResult:
        result = MatchResult()
        imports = context.graph_or_function.opset_imports
        if imports.get("", context.model.opset_imports.get("", 0)) < 15:
            return result.fail("Shape start/end attributes require opset 15.")
        if x.shape is None:
            return result.fail("Input rank is needed to validate Gather indices.")
        axis = context.root.attributes.get("axis")
        if axis is not None and (axis.is_ref() or axis.as_int() not in (0, -1)):
            return result.fail("Gather must operate along its one-dimensional shape input.")
        shape_node = context.root.inputs[0].producer()
        if _shape_attributes(shape_node) is None:
            return result.fail("Shape attributes refer to function parameters.")
        array = get_numpy_value(indices)
        if array is None or array.ndim > 1 or array.dtype not in (np.int32, np.int64):
            return result.fail("Gather indices must be a constant integer scalar or vector.")
        dims = _shape_indices(shape_node, len(x.shape))
        normalized = [int(i) + len(dims) if i < 0 else int(i) for i in array.flat]
        if any(i < 0 or i >= len(dims) for i in normalized):
            return result.fail("Gather indices are out of range.")
        selected = [dims[i] for i in normalized]
        if selected and selected != list(range(selected[0], selected[0] + len(selected))):
            return result.fail("Gather indices do not select contiguous dimensions.")
        self._selected = selected
        self._scalar = array.ndim == 0
        return result

    def rewrite(self, op, x, indices):
        if not self._selected:
            return op.Constant(value=ir.tensor([], dtype=ir.DataType.INT64))
        shape = op.Shape(x, start=self._selected[0], end=self._selected[-1] + 1)
        if self._scalar:
            axes = op.Constant(value=ir.tensor([0], dtype=ir.DataType.INT64))
            return op.Squeeze(shape, axes)
        return shape


class ShapeUnsqueeze(RewriteRuleClassBase):
    def pattern(self, op, x, axes):
        return op.Shape(op.Unsqueeze(x, axes))

    def check(self, context, x, axes) -> MatchResult:
        result = MatchResult()
        if x.shape is None:
            return result.fail("Input rank is needed to resolve inserted dimensions.")
        array = get_numpy_value(axes)
        if array is None or array.ndim != 1 or array.dtype != np.int64:
            return result.fail("Unsqueeze axes must be a constant int64 vector.")
        rank = len(x.shape) + array.size
        normalized = [int(i) + rank if i < 0 else int(i) for i in array]
        if len(set(normalized)) != len(normalized) or any(
            i < 0 or i >= rank for i in normalized
        ):
            return result.fail("Unsqueeze axes are invalid.")
        if _shape_attributes(context.root) is None:
            return result.fail("Shape attributes refer to function parameters.")
        source = iter(range(len(x.shape)))
        dims = [None if i in normalized else next(source) for i in range(rank)]
        self._dims = [dims[i] for i in _shape_indices(context.root, rank)]
        return result

    def rewrite(self, op, x, axes):
        if not self._dims:
            return op.Constant(value=ir.tensor([], dtype=ir.DataType.INT64))
        shape = None
        parts = []
        cursor = 0
        while cursor < len(self._dims):
            inserted = self._dims[cursor] is None
            end = cursor + 1
            while end < len(self._dims) and (self._dims[end] is None) == inserted:
                end += 1
            if inserted:
                parts.append(
                    op.Constant(value=ir.tensor([1] * (end - cursor), dtype=ir.DataType.INT64))
                )
            else:
                if shape is None:
                    shape = op.Shape(x)
                indices = op.Constant(
                    value=ir.tensor(self._dims[cursor:end], dtype=ir.DataType.INT64)
                )
                parts.append(op.Gather(shape, indices, axis=0))
            cursor = end
        return op.Concat(*parts, axis=0) if len(parts) > 1 else op.Identity(parts[0])


class SameShapeChildren(RewriteRuleClassBase):
    """Share layout results even when their shape expressions differ."""

    def __init__(self, op_type: str):
        self.op_type = op_type
        super().__init__(f"SameShape{op_type}")

    def pattern(self, op, x, target):
        return getattr(op, self.op_type)(x, target)

    def check(self, context, x, target) -> MatchResult:
        result = MatchResult()
        root = context.root
        output = root.outputs[0]
        for node in root.graph:
            if node is root:
                break
            if (
                node.domain == root.domain
                and node.op_type == self.op_type
                and node.inputs[0] is x
                and same_shape(output.shape, node.outputs[0].shape)
            ):
                self._existing = node.outputs[0]
                return result
        return result.fail("No earlier layout result has the same proven shape.")

    def rewrite(self, op, x, target):
        return op.Identity(self._existing)


class SimplifyExpandShape(RewriteRuleClassBase):
    """Avoid computing target dimensions already supplied by the data input."""

    def pattern(self, op, x, target):
        return op.Expand(x, target)

    def check(self, context, x, target) -> MatchResult:
        result = MatchResult()
        output = context.root.outputs[0]
        concat = target.producer()
        if (
            x.shape is None
            or output.shape is None
            or len(x.shape) != len(output.shape)
            or concat is None
            or concat.domain != ""
            or concat.op_type != "Concat"
            or len(concat.inputs) != len(x.shape)
        ):
            return result.fail("Expand must have a rank-aligned concatenated shape.")
        axis = concat.attributes.get("axis")
        if axis is None or axis.is_ref() or axis.as_int() not in (0, -1):
            return result.fail("Expand shape must concatenate dimension vectors.")
        equal = [same_dim(a, b) for a, b in zip(x.shape, output.shape)]
        if sum(not item for item in equal) != 1:
            return result.fail("Exactly one output dimension must differ.")
        changed = False
        self._parts = []
        for part, unchanged in zip(concat.inputs, equal):
            if part is None or part.shape is None or list(part.shape) != [1]:
                return result.fail("Each concatenated input must describe one dimension.")
            if unchanged:
                value = get_numpy_value(part)
                changed |= value is None or value.size != 1 or int(value.item()) != 1
                self._parts.append(None)
            else:
                self._parts.append(part)
        if not changed:
            return result.fail("Unchanged dimensions already use unit broadcast targets.")
        return result

    def rewrite(self, op, x, target):
        parts = [
            op.Constant(value=ir.tensor([1], dtype=ir.DataType.INT64))
            if part is None
            else part
            for part in self._parts
        ]
        return op.Expand(x, op.Concat(*parts, axis=0))


shape_cast_rule = ShapeCast.rule(remove_nodes=False)
shape_transpose_rule = ShapeTranspose.rule(remove_nodes=False)
gather_shape_rule = GatherShape.rule(remove_nodes=False)
shape_unsqueeze_rule = ShapeUnsqueeze.rule(remove_nodes=False)
same_shape_expand_rule = SameShapeChildren.rule("Expand")
same_shape_reshape_rule = SameShapeChildren.rule("Reshape")
simplify_expand_shape_rule = SimplifyExpandShape.rule(remove_nodes=False)

rules = RewriteRuleSet(
    [
        shape_cast_rule,
        shape_transpose_rule,
        gather_shape_rule,
        shape_unsqueeze_rule,
        same_shape_expand_rule,
        same_shape_reshape_rule,
        simplify_expand_shape_rule,
    ]
)
