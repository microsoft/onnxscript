# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optimizations for ONNX collection operators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

_INT64_MAX = np.iinfo(np.int64).max


def _ints(value: ir.Value | None, *, ranks: Sequence[int] = (0, 1)) -> np.ndarray | None:
    array = _ir_utils.get_numpy_value(value)
    if array is None or array.ndim not in ranks or array.dtype not in (np.int32, np.int64):
        return None
    return array


def _axis(attr: ir.Attr, rank: int | None) -> int | None:
    if attr.is_ref():
        return None
    value = attr.as_int()
    if value < 0:
        if rank is None:
            return None
        value += rank
    if rank is not None and not 0 <= value < rank:
        return None
    return value


def _node_axis(node: ir.Node, rank: int | None, default: int = 0) -> int | None:
    attr = node.attributes.get("axis")
    if attr is None:
        value = default
        if value < 0:
            if rank is None:
                return None
            value += rank
        return value if rank is None or 0 <= value < rank else None
    return _axis(attr, rank)


def _default_opset(context) -> int | None:
    imports = context.graph_or_function.opset_imports
    return imports.get("", imports.get("ai.onnx"))


def _concrete_attributes(node: ir.Node) -> dict[str, ir.Attr] | None:
    if any(attribute.is_ref() for attribute in node.attributes.values()):
        return None
    return dict(node.attributes)


def _const(op, values: Sequence[int], *, scalar: bool = False):
    array = np.asarray(values[0] if scalar else values, dtype=np.int64)
    return op.Constant(value=ir.tensor(array))


def _exclusive_intermediate(context, value: ir.Value) -> bool:
    return len(value.uses()) == 1 and value not in set(context.graph_or_function.outputs)


_ELEMENTWISE_UNARY_OPS = (
    "Abs",
    "Ceil",
    "Cos",
    "Erf",
    "Exp",
    "Floor",
    "Neg",
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


class RemoveEmptyConcatInputs(RewriteRuleClassBase):
    """Remove statically empty inputs from an arbitrary-arity Concat."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, first, axis):
        return op.Concat(first, axis=axis, _allow_other_inputs=True)

    def check(self, context, first, axis) -> MatchResult:
        del first
        result = MatchResult()
        values = [value for value in context.root.inputs if value is not None]
        if len(values) < 2:
            return result.fail("Concat must have at least two inputs.")
        rank = values[0].shape.rank() if values[0].shape is not None else None
        normalized = _axis(axis, rank)
        if normalized is None:
            return result.fail("Concat axis cannot be normalized.")
        self._kept = []
        for value in values:
            if value.shape is None or value.shape.rank() != rank:
                self._kept.append(value)
                continue
            dim = value.shape[normalized]
            if not isinstance(dim, int) or dim != 0:
                self._kept.append(value)
        if len(self._kept) == len(values):
            return result.fail("No input is statically empty along the Concat axis.")
        if not self._kept:
            return result.fail(
                "Removing every input would lose the output type/shape contract."
            )
        self._concat_axis = axis.as_int()
        return result

    def rewrite(self, op, first, axis):
        del first, axis
        if len(self._kept) == 1:
            return op.Identity(self._kept[0])
        return op.Concat(*self._kept, axis=self._concat_axis)


class ComposeConsecutiveGathers(RewriteRuleClassBase):
    """Compose same-axis Gathers when the inner indices are a constant vector."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, x, inner_indices, outer_indices):
        inner = op.Gather(x, inner_indices)
        return op.Gather(inner, outer_indices)

    def check(
        self,
        context,
        x,
        inner_indices,
        outer_indices,
    ) -> MatchResult:
        result = MatchResult()
        inner = _ints(inner_indices, ranks=(1,))
        outer = _ints(outer_indices)
        if inner is None or outer is None:
            return result.fail("Gather indices must be constant integer tensors.")
        rank = x.shape.rank() if x.shape is not None else None
        gather_nodes = [node for node in context.nodes if node.op_type == "Gather"]
        inner_node = next(node for node in gather_nodes if node.inputs[0] is x)
        outer_node = next(node for node in gather_nodes if node is not inner_node)
        inner_axis = _node_axis(inner_node, rank)
        outer_axis = _node_axis(outer_node, rank)
        if inner_axis is None or outer_axis is None or inner_axis != outer_axis:
            return result.fail("Consecutive Gather axes differ or cannot be normalized.")
        if x.shape is None or not isinstance(x.shape[inner_axis], int):
            return result.fail("The gathered input dimension must be static.")
        input_dim = x.shape[inner_axis]
        normalized_inner = np.where(inner < 0, inner + input_dim, inner)
        if np.any(normalized_inner < 0) or np.any(normalized_inner >= input_dim):
            return result.fail("Inner Gather index is out of range.")
        length = inner.size
        normalized = np.where(outer < 0, outer + length, outer)
        if np.any(normalized < 0) or np.any(normalized >= length):
            return result.fail("Outer Gather index is out of range.")
        self._indices = inner[normalized].astype(inner.dtype, copy=False)
        self._axis = inner_axis
        return result

    def rewrite(self, op, x, inner_indices, outer_indices):
        del inner_indices, outer_indices
        return op.Gather(x, op.Constant(value=ir.tensor(self._indices)), axis=self._axis)


class ReplaceArithmeticProgressionGatherWithSlice(RewriteRuleClassBase):
    """Replace a constant arithmetic-progression Gather by Slice."""

    def pattern(self, op, x, indices):
        return op.Gather(x, indices)

    def check(self, context, x, indices) -> MatchResult:
        result = MatchResult()
        values = _ints(indices)
        if values is None or values.size == 0:
            return result.fail("Indices must be a non-empty constant scalar or vector.")
        opset = _default_opset(context)
        if opset is not None and opset < 10:
            return result.fail("Input-form Slice requires opset 10 or later.")
        if values.ndim == 0 and opset is not None and opset < 13:
            return result.fail("Input-form Squeeze requires opset 13 or later.")
        rank = x.shape.rank() if x.shape is not None else None
        normalized_axis = _node_axis(context.root, rank)
        if normalized_axis is None:
            return result.fail("Gather axis cannot be normalized.")
        flat = values.reshape(-1).astype(np.int64)
        if x.shape is None or not isinstance(x.shape[normalized_axis], int):
            return result.fail("The gathered axis dimension must be static.")
        dim = x.shape[normalized_axis]
        if np.any(flat < 0):
            flat = np.where(flat < 0, flat + dim, flat)
        if np.any(flat < 0) or np.any(flat >= dim):
            return result.fail("Gather index is out of range.")
        step = 1
        if flat.size > 1:
            diffs = np.diff(flat)
            if diffs[0] <= 0 or np.any(diffs != diffs[0]):
                return result.fail(
                    "Indices are not a strictly ascending arithmetic progression."
                )
            step = int(diffs[0])
        self._start = int(flat[0])
        self._end = self._start + step * flat.size
        self._step = step
        self._axis = normalized_axis
        self._scalar = values.ndim == 0
        return result

    def rewrite(self, op, x, indices):
        del indices
        sliced = op.Slice(
            x,
            _const(op, [self._start]),
            _const(op, [self._end]),
            _const(op, [self._axis]),
            _const(op, [self._step]),
        )
        if self._scalar:
            return op.Squeeze(sliced, _const(op, [self._axis]))
        return sliced


class SelectConcatInputWithSlice(RewriteRuleClassBase):
    """Route an exact one-axis Slice of a Concat to the selected operand."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, data, starts, ends, axes, steps):
        return op.Slice(data, starts, ends, axes, steps)

    def check(self, context, data, starts, ends, axes, steps) -> MatchResult:
        result = MatchResult()
        concat = data.producer()
        if concat is None or concat.domain != "" or concat.op_type != "Concat":
            return result.fail("Slice input is not produced by a default-domain Concat.")
        start = _ints(starts, ranks=(1,))
        end = _ints(ends, ranks=(1,))
        slice_axes = _ints(axes, ranks=(1,))
        slice_steps = _ints(steps, ranks=(1,))
        if any(
            value is None or value.size != 1 for value in (start, end, slice_axes, slice_steps)
        ):
            return result.fail("Slice must have constant singleton parameters.")
        if int(slice_steps.item()) != 1:
            return result.fail("Slice step must be one.")
        if not concat.inputs:
            return result.fail("Concat has no inputs.")
        first = concat.inputs[0]
        rank = first.shape.rank() if first is not None and first.shape is not None else None
        concat_axis_attr = concat.attributes.get("axis")
        if concat_axis_attr is None:
            return result.fail("Concat has no axis attribute.")
        concat_axis = _axis(concat_axis_attr, rank)
        raw_slice_axis = int(slice_axes.item())
        slice_axis = (
            raw_slice_axis + rank
            if rank is not None and raw_slice_axis < 0
            else raw_slice_axis
        )
        if concat_axis is None or slice_axis != concat_axis:
            return result.fail("Slice and Concat axes differ or cannot be normalized.")
        sizes: list[int] = []
        for value in concat.inputs:
            if value is None or value.shape is None or value.shape.rank() != rank:
                return result.fail("Concat input ranks must be known and equal.")
            dim = value.shape[concat_axis]
            if not isinstance(dim, int):
                return result.fail("Concat-axis dimensions must be static.")
            sizes.append(dim)
        total = sum(sizes)
        begin = int(start.item())
        finish = int(end.item())
        begin = begin + total if begin < 0 else begin
        finish = finish + total if finish < 0 and finish != _INT64_MAX else finish
        begin = min(max(begin, 0), total)
        finish = total if finish == _INT64_MAX else min(max(finish, 0), total)
        offset = 0
        for value, size in zip(concat.inputs, sizes):
            if begin == offset and finish == offset + size:
                assert value is not None
                self._selected = value
                return result
            offset += size
        return result.fail("Slice does not exactly recover one Concat input.")

    def rewrite(self, op, **_):
        return op.Identity(self._selected)


class MoveUnaryBeforeDuplicateConcat(RewriteRuleClassBase):
    """Push an attribute-free elementwise unary op through Concat(x, x)."""

    def __init__(self, op_type: str):
        super().__init__(f"Move{op_type}BeforeDuplicateConcat", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, axis):
        joined = op.Concat(x, x, axis=axis)
        return getattr(op, self._op_type)(joined)

    def check(self, context, x, axis):
        del context, x
        result = MatchResult()
        if axis.is_ref():
            return result.fail("Reference attributes are not rewritten.")
        return result

    def rewrite(self, op, x, axis):
        transformed = getattr(op, self._op_type)(x)
        return op.Concat(transformed, transformed, axis=axis)


class MoveCastBeforeDuplicateConcat(RewriteRuleClassBase):
    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, x, axis):
        return op.Cast(op.Concat(x, x, axis=axis))

    def check(self, context, x, axis):
        del x
        result = MatchResult()
        if axis.is_ref():
            return result.fail("Reference attributes are not rewritten.")
        attributes = _concrete_attributes(context.root)
        if attributes is None:
            return result.fail("Cast reference attributes cannot be moved.")
        if "to" not in attributes:
            return result.fail("Cast is missing its required 'to' attribute.")
        self._cast_attrs = attributes
        return result

    def rewrite(self, op, x, axis):
        cast = op.Cast(x, **self._cast_attrs)
        return op.Concat(cast, cast, axis=axis)


class MoveScalarBinaryBeforeDuplicateConcat(RewriteRuleClassBase):
    """Push an elementwise binary op with a scalar RHS through Concat(x, x)."""

    def __init__(self, op_type: str):
        super().__init__(f"MoveScalar{op_type}BeforeDuplicateConcat", remove_nodes=False)
        self._op_type = op_type

    def pattern(self, op, x, scalar, axis):
        joined = op.Concat(x, x, axis=axis)
        return getattr(op, self._op_type)(joined, scalar)

    def check(self, context, x, scalar, axis) -> MatchResult:
        del context, x
        result = MatchResult()
        if axis.is_ref():
            return result.fail("Reference attributes are not rewritten.")
        value = _ir_utils.get_numpy_value(scalar)
        if value is None or value.ndim != 0:
            return result.fail("The binary RHS must be a constant scalar.")
        return result

    def rewrite(self, op, x, scalar, axis):
        transformed = getattr(op, self._op_type)(x, scalar)
        return op.Concat(transformed, transformed, axis=axis)


class MoveUnaryBeforeGather(RewriteRuleClassBase):
    """Move a constant scalar-or-vector Gather before an elementwise unary op."""

    def __init__(self, op_type: str):
        super().__init__(f"Move{op_type}BeforeGather")
        self._op_type = op_type

    def pattern(self, op, x, indices):
        produced = getattr(op, self._op_type)(x)
        return op.Gather(produced, indices)

    def check(self, context, x, indices) -> MatchResult:
        result = MatchResult()
        if _ints(indices, ranks=(0, 1)) is None:
            return result.fail("Gather indices must be a constant scalar or vector.")
        produced = next(
            node for node in context.nodes if node.op_type == self._op_type
        ).outputs[0]
        if not _exclusive_intermediate(context, produced):
            return result.fail("Producer output is shared or is a graph output.")
        rank = x.shape.rank() if x.shape is not None else None
        gather_axis = _node_axis(context.root, rank)
        if gather_axis is None:
            return result.fail("Gather axis cannot be normalized.")
        self._gather_axis = gather_axis
        return result

    def rewrite(self, op, x, indices):
        gathered = op.Gather(x, indices, axis=self._gather_axis)
        return getattr(op, self._op_type)(gathered)


class MoveCastBeforeGather(RewriteRuleClassBase):
    def pattern(self, op, x, indices):
        return op.Gather(op.Cast(x), indices)

    def check(self, context, x, indices) -> MatchResult:
        result = MatchResult()
        if _ints(indices, ranks=(0, 1)) is None:
            return result.fail("Gather indices must be a constant scalar or vector.")
        if x.dtype is None or x.dtype == ir.DataType.STRING:
            return result.fail(
                "Cast cannot move after Gather for string or unknown input types."
            )
        cast_node = next(node for node in context.nodes if node.op_type == "Cast")
        if not _exclusive_intermediate(context, cast_node.outputs[0]):
            return result.fail("Cast output is shared or is a graph output.")
        attributes = _concrete_attributes(cast_node)
        if attributes is None:
            return result.fail("Cast reference attributes cannot be moved.")
        if "to" not in attributes:
            return result.fail("Cast is missing its required 'to' attribute.")
        self._cast_attrs = attributes
        rank = x.shape.rank() if x.shape is not None else None
        gather_axis = _node_axis(context.root, rank)
        if gather_axis is None:
            return result.fail("Gather axis cannot be normalized.")
        self._gather_axis = gather_axis
        return result

    def rewrite(self, op, x, indices):
        return op.Cast(op.Gather(x, indices, axis=self._gather_axis), **self._cast_attrs)


class MoveBroadcastBinaryBeforeGather(RewriteRuleClassBase):
    """Move a vector Gather ahead of a broadcast-compatible pointwise binary op."""

    def __init__(self, op_type: str):
        super().__init__(f"MoveBroadcast{op_type}BeforeGather")
        self._op_type = op_type

    def pattern(self, op, lhs, rhs, indices):
        produced = getattr(op, self._op_type)(lhs, rhs)
        return op.Gather(produced, indices)

    def check(self, context, lhs, rhs, indices) -> MatchResult:
        result = MatchResult()
        index_array = _ints(indices, ranks=(1,))
        if index_array is None:
            return result.fail("Only rank-preserving constant vector Gather is supported.")
        producer = next(node for node in context.nodes if node.op_type == self._op_type)
        output = producer.outputs[0]
        if not _exclusive_intermediate(context, output):
            return result.fail("Producer output must be exclusive.")
        if lhs.shape is None or rhs.shape is None:
            return result.fail("Operand shapes must be known.")
        output_rank = max(lhs.shape.rank(), rhs.shape.rank())
        normalized_axis = _node_axis(context.root, output_rank)
        if normalized_axis is None:
            return result.fail("Gather axis cannot be normalized.")
        aligned_dims: list[ir.SymbolicDim | int | None] = []
        for operand in (lhs, rhs):
            gap = output_rank - operand.shape.rank()
            aligned_dims.append(
                None if normalized_axis < gap else operand.shape[normalized_axis - gap]
            )
        output_dim = next(
            (
                dim
                for dim in aligned_dims
                if dim is not None and not (isinstance(dim, int) and dim == 1)
            ),
            1,
        )
        for dim in aligned_dims:
            if dim is None or (isinstance(dim, int) and dim == 1):
                continue
            if dim != output_dim:
                return result.fail("Operands are not provably broadcast-compatible.")
        self._operand_axes: list[int | None] = []
        for operand in (lhs, rhs):
            gap = output_rank - operand.shape.rank()
            if normalized_axis < gap:
                self._operand_axes.append(None)
                continue
            local_axis = normalized_axis - gap
            operand_dim = operand.shape[local_axis]
            if isinstance(operand_dim, int) and operand_dim == 1:
                self._operand_axes.append(None)
            elif operand_dim == output_dim and (
                isinstance(operand_dim, int)
                or (isinstance(operand_dim, ir.SymbolicDim) and operand_dim.value is not None)
            ):
                self._operand_axes.append(local_axis)
            else:
                return result.fail("Operand dimension is not provably equal or broadcast-one.")
        if all(axis is None for axis in self._operand_axes):
            return result.fail("No operand carries the gathered dimension.")
        return result

    def rewrite(self, op, lhs, rhs, indices):
        operands = [lhs, rhs]
        for i, local_axis in enumerate(self._operand_axes):
            if local_axis is not None:
                operands[i] = op.Gather(operands[i], indices, axis=local_axis)
        return getattr(op, self._op_type)(*operands)


class SelectConcatInputWithGather(RewriteRuleClassBase):
    """Route a constant Gather into one input of an arbitrary-arity Concat."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, data, indices):
        return op.Gather(data, indices)

    def check(self, context, data, indices) -> MatchResult:
        result = MatchResult()
        concat = data.producer()
        if concat is None or concat.domain != "" or concat.op_type != "Concat":
            return result.fail("Gather input is not produced by a default-domain Concat.")
        values = list(concat.inputs)
        if len(values) < 2 or any(value is None for value in values):
            return result.fail("Concat must have at least two present inputs.")
        rank = values[0].shape.rank() if values[0].shape is not None else None
        concat_axis_attr = concat.attributes.get("axis")
        if concat_axis_attr is None:
            return result.fail("Concat has no axis attribute.")
        concat_axis = _axis(concat_axis_attr, rank)
        gather_axis = _node_axis(context.root, rank)
        if concat_axis is None or gather_axis is None or concat_axis != gather_axis:
            return result.fail("Gather and Concat axes differ or cannot be normalized.")
        index_array = _ints(indices, ranks=(0, 1))
        if index_array is None or index_array.size == 0:
            return result.fail("Gather indices must be a non-empty constant scalar or vector.")
        sizes: list[int] = []
        for value in values:
            assert value is not None
            if value.shape is None or value.shape.rank() != rank:
                return result.fail("Every Concat input rank must be known and equal.")
            dim = value.shape[concat_axis]
            if not isinstance(dim, int):
                return result.fail("Concat-axis dimensions must be static.")
            sizes.append(dim)
        flat = index_array.reshape(-1).astype(np.int64)
        total = sum(sizes)
        flat = np.where(flat < 0, flat + total, flat)
        if np.any(flat < 0) or np.any(flat >= total):
            return result.fail("Gather index is out of range.")
        offset = 0
        for value, size in zip(values, sizes):
            if np.all((flat >= offset) & (flat < offset + size)):
                assert value is not None
                self._selected = value
                self._local_indices = (
                    (flat - offset)
                    .reshape(index_array.shape)
                    .astype(index_array.dtype, copy=False)
                )
                self._gather_axis = gather_axis
                return result
            offset += size
        return result.fail("Gather indices span more than one Concat input.")

    def rewrite(self, op, data, indices):
        del data, indices
        local = op.Constant(value=ir.tensor(self._local_indices))
        return op.Gather(self._selected, local, axis=self._gather_axis)


class FuseConsecutiveSlices(RewriteRuleClassBase):
    """Fuse consecutive constant Slices on disjoint axes."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(
        self,
        op,
        x,
        starts0,
        ends0,
        axes0,
        steps0,
        starts1,
        ends1,
        axes1,
        steps1,
    ):
        first = op.Slice(x, starts0, ends0, axes0, steps0)
        return op.Slice(first, starts1, ends1, axes1, steps1)

    def check(
        self,
        context,
        x,
        starts0,
        ends0,
        axes0,
        steps0,
        starts1,
        ends1,
        axes1,
        steps1,
    ) -> MatchResult:
        del context
        result = MatchResult()
        arrays = [
            _ints(v, ranks=(1,))
            for v in (starts0, ends0, axes0, steps0, starts1, ends1, axes1, steps1)
        ]
        if any(v is None for v in arrays):
            return result.fail("Slice parameters must be constant integer vectors.")
        s0, e0, a0, st0, s1, e1, a1, st1 = arrays
        assert all(v is not None for v in arrays)
        if not (len(s0) == len(e0) == len(a0) == len(st0)):
            return result.fail("First Slice parameter lengths differ.")
        if not (len(s1) == len(e1) == len(a1) == len(st1)):
            return result.fail("Second Slice parameter lengths differ.")
        rank = x.shape.rank() if x.shape is not None else None
        if rank is None and (np.any(a0 < 0) or np.any(a1 < 0)):
            return result.fail("Negative axes require a known input rank.")
        na0 = np.where(a0 < 0, a0 + rank, a0) if rank is not None else a0
        na1 = np.where(a1 < 0, a1 + rank, a1) if rank is not None else a1
        if rank is not None and (
            np.any(na0 < 0) or np.any(na0 >= rank) or np.any(na1 < 0) or np.any(na1 >= rank)
        ):
            return result.fail("Slice axis is out of range.")
        if len(set(na0.tolist())) != len(na0) or len(set(na1.tolist())) != len(na1):
            return result.fail("Slice axes must be unique.")
        if set(na0.tolist()) & set(na1.tolist()):
            return result.fail("Slices operate on a common axis.")
        self._starts = np.concatenate((s0, s1)).astype(np.int64)
        self._ends = np.concatenate((e0, e1)).astype(np.int64)
        self._axes = np.concatenate((na0, na1)).astype(np.int64)
        self._steps = np.concatenate((st0, st1)).astype(np.int64)
        return result

    def rewrite(self, op, x, **_):
        return op.Slice(
            x,
            op.Constant(value=ir.tensor(self._starts)),
            op.Constant(value=ir.tensor(self._ends)),
            op.Constant(value=ir.tensor(self._axes)),
            op.Constant(value=ir.tensor(self._steps)),
        )


class FuseSpatialSlicesIntoSpaceToDepth(RewriteRuleClassBase):
    """Recognize the canonical four spatial phases when the channel count is one."""

    def pattern(
        self,
        op,
        x,
        s0,
        e0,
        a0,
        st0,
        s1,
        e1,
        a1,
        st1,
        s2,
        e2,
        a2,
        st2,
        s3,
        e3,
        a3,
        st3,
        concat_axis,
    ):
        y0 = op.Slice(x, s0, e0, a0, st0)
        y1 = op.Slice(x, s1, e1, a1, st1)
        y2 = op.Slice(x, s2, e2, a2, st2)
        y3 = op.Slice(x, s3, e3, a3, st3)
        return op.Concat(y0, y1, y2, y3, axis=concat_axis)

    def check(self, context, x, concat_axis, **params) -> MatchResult:
        del context
        result = MatchResult()
        if x.shape is None or x.shape.rank() != 4:
            return result.fail("SpaceToDepth input must have rank four.")
        if x.shape[1] != 1:
            return result.fail("Phase-major Concat equals DCR SpaceToDepth only for C=1.")
        if not isinstance(x.shape[2], int) or not isinstance(x.shape[3], int):
            return result.fail("Spatial dimensions must be static.")
        if x.shape[2] % 2 or x.shape[3] % 2:
            return result.fail("Spatial dimensions must be even.")
        if _axis(concat_axis, 4) != 1:
            return result.fail("Concat must use the channel axis.")
        expected = ((0, 0), (0, 1), (1, 0), (1, 1))
        for i, phase in enumerate(expected):
            starts = _ints(params[f"s{i}"], ranks=(1,))
            ends = _ints(params[f"e{i}"], ranks=(1,))
            axes = _ints(params[f"a{i}"], ranks=(1,))
            steps = _ints(params[f"st{i}"], ranks=(1,))
            if any(value is None or value.size != 2 for value in (starts, ends, axes, steps)):
                return result.fail("Each phase must be a constant two-axis Slice.")
            normalized_axes = np.where(axes < 0, axes + 4, axes)
            if normalized_axes.tolist() != [2, 3]:
                return result.fail("Each phase must slice H and W in that order.")
            if starts.tolist() != list(phase) or steps.tolist() != [2, 2]:
                return result.fail("Slice does not select the expected spatial phase.")
            valid_ends = int(ends[0]) in (_INT64_MAX, x.shape[2]) and int(ends[1]) in (
                _INT64_MAX,
                x.shape[3],
            )
            if not valid_ends:
                return result.fail("Slice crops instead of spanning the spatial dimensions.")
        return result

    def rewrite(self, op, x, **_):
        return op.SpaceToDepth(x, blocksize=2)


def _check_complete_gather_partition(
    rule,
    context,
    x: ir.Value,
    index_values: Sequence[ir.Value],
) -> MatchResult:
    result = MatchResult()
    opset = _default_opset(context)
    if opset is not None and opset < 13:
        return result.fail("Input-form Split requires opset 13 or later.")
    indices = [_ints(index) for index in index_values]
    if any(index is None or index.size != 1 for index in indices):
        return result.fail("Gather indices must be constant scalars or singleton vectors.")
    assert all(index is not None for index in indices)
    if any(index.ndim != indices[0].ndim for index in indices[1:]):
        return result.fail("Gather index ranks must match.")
    rank = x.shape.rank() if x.shape is not None else None
    gather_nodes = [value.producer() for value in context.output_values]
    if any(node is None for node in gather_nodes):
        return result.fail("Matched Gather output has no producer.")
    axes = [_node_axis(node, rank) for node in gather_nodes if node is not None]
    axis = axes[0]
    if axis is None or any(item != axis for item in axes[1:]):
        return result.fail("Gather axes differ or cannot be normalized.")
    count = len(indices)
    if x.shape is None or x.shape[axis] != count:
        return result.fail("Gathered axis must match the number of Gathers.")
    slots = [int(index.item()) for index in indices]
    slots = [slot + count if slot < 0 else slot for slot in slots]
    if sorted(slots) != list(range(count)):
        return result.fail("Gathers do not partition the complete axis.")
    rule._slots = slots
    rule._axis = axis
    rule._scalar = indices[0].ndim == 0
    return result


def _replace_gather_partition_with_split(rule, op, x: ir.Value):
    count = len(rule._slots)
    splits = op.Split(x, _const(op, [1] * count), axis=rule._axis, _outputs=count)
    outputs = [splits[slot] for slot in rule._slots]
    if rule._scalar:
        axes = _const(op, [rule._axis])
        outputs = [op.Squeeze(output, axes) for output in outputs]
    return tuple(outputs)


def _gather_index_selects(expected: int):
    def check(context, node) -> bool:
        del context
        index = _ints(node.inputs[1], ranks=(0, 1))
        data = node.inputs[0]
        if index is None or index.size != 1 or data is None or data.shape is None:
            return False
        axis = _node_axis(node, data.shape.rank())
        if axis is None or not isinstance(data.shape[axis], int):
            return False
        value = int(index.item())
        if value < 0:
            value += data.shape[axis]
        return value == expected

    return check


class ReplaceGatherPartitionWithSplit2(RewriteRuleClassBase):
    """Replace two sibling Gathers that completely partition an axis with Split."""

    def pattern(self, op, x, i0, i1):
        return (
            op.Gather(x, i0, _check=_gather_index_selects(0)),
            op.Gather(x, i1, _check=_gather_index_selects(1)),
        )

    def check(self, context, x, i0, i1) -> MatchResult:
        return _check_complete_gather_partition(self, context, x, (i0, i1))

    def rewrite(self, op, x, **_):
        return _replace_gather_partition_with_split(self, op, x)


class ReplaceGatherPartitionWithSplit3(RewriteRuleClassBase):
    """Replace three sibling Gathers that completely partition an axis with Split."""

    def pattern(self, op, x, i0, i1, i2):
        return (
            op.Gather(x, i0, _check=_gather_index_selects(0)),
            op.Gather(x, i1, _check=_gather_index_selects(1)),
            op.Gather(x, i2, _check=_gather_index_selects(2)),
        )

    def check(self, context, x, i0, i1, i2) -> MatchResult:
        return _check_complete_gather_partition(self, context, x, (i0, i1, i2))

    def rewrite(self, op, x, **_):
        return _replace_gather_partition_with_split(self, op, x)


class ReplaceGatherPartitionWithSplit4(RewriteRuleClassBase):
    """Replace four sibling Gathers that completely partition an axis with Split."""

    def pattern(self, op, x, i0, i1, i2, i3):
        return (
            op.Gather(x, i0, _check=_gather_index_selects(0)),
            op.Gather(x, i1, _check=_gather_index_selects(1)),
            op.Gather(x, i2, _check=_gather_index_selects(2)),
            op.Gather(x, i3, _check=_gather_index_selects(3)),
        )

    def check(self, context, x, i0, i1, i2, i3) -> MatchResult:
        return _check_complete_gather_partition(self, context, x, (i0, i1, i2, i3))

    def rewrite(self, op, x, **_):
        return _replace_gather_partition_with_split(self, op, x)


def _read_slice_range(
    x: ir.Value,
    starts: ir.Value,
    ends: ir.Value,
    axes: ir.Value,
    steps: ir.Value,
) -> tuple[int, int, int] | None:
    if x.shape is None:
        return None
    start = _ints(starts, ranks=(1,))
    end = _ints(ends, ranks=(1,))
    axis_array = _ints(axes, ranks=(1,))
    step = _ints(steps, ranks=(1,))
    if any(value is None or value.size != 1 for value in (start, end, axis_array, step)):
        return None
    if int(step.item()) != 1:
        return None
    rank = x.shape.rank()
    axis = int(axis_array.item())
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank or not isinstance(x.shape[axis], int):
        return None
    dim = x.shape[axis]
    begin = int(start.item())
    finish = int(end.item())
    begin = begin + dim if begin < 0 else begin
    finish = finish + dim if finish < 0 and finish != _INT64_MAX else finish
    begin = min(max(begin, 0), dim)
    finish = dim if finish == _INT64_MAX else min(max(finish, 0), dim)
    if begin >= finish:
        return None
    return axis, begin, finish


def _check_complete_slice_partition(rule, context, x, parameters) -> MatchResult:
    result = MatchResult()
    opset = _default_opset(context)
    if opset is not None and opset < 13:
        return result.fail("Input-form Split requires opset 13 or later.")
    ranges = [
        _read_slice_range(x, starts, ends, axes, steps)
        for starts, ends, axes, steps in parameters
    ]
    if any(item is None for item in ranges):
        return result.fail("Slices must have valid constant singleton parameters.")
    assert all(item is not None for item in ranges)
    if any(item[0] != ranges[0][0] for item in ranges[1:]):
        return result.fail("Slices use different axes.")
    ordered = sorted(enumerate(ranges), key=lambda item: item[1][1])
    axis = ranges[0][0]
    dim = x.shape[axis]
    assert isinstance(dim, int)
    expected = 0
    for _, item in ordered:
        if item[1] != expected:
            return result.fail("Slice ranges are not a contiguous partition.")
        expected = item[2]
    if expected != dim:
        return result.fail("Slice ranges do not cover the complete axis.")
    rule._sizes = [item[2] - item[1] for _, item in ordered]
    rule._output_slots = [0] * len(ordered)
    for split_slot, (original_slot, _) in enumerate(ordered):
        rule._output_slots[original_slot] = split_slot
    rule._axis = axis
    return result


def _replace_slice_partition_with_split(rule, op, x):
    outputs = op.Split(
        x,
        _const(op, rule._sizes),
        axis=rule._axis,
        _outputs=len(rule._sizes),
    )
    return tuple(outputs[slot] for slot in rule._output_slots)


class ReplaceSlicePartitionWithSplit2(RewriteRuleClassBase):
    """Replace two constant sibling Slices covering an axis with Split."""

    def pattern(self, op, x, s0, e0, a0, st0, s1, e1, a1, st1):
        return (
            op.Slice(x, s0, e0, a0, st0),
            op.Slice(x, s1, e1, a1, st1),
        )

    def check(self, context, x, s0, e0, a0, st0, s1, e1, a1, st1) -> MatchResult:
        return _check_complete_slice_partition(
            self,
            context,
            x,
            ((s0, e0, a0, st0), (s1, e1, a1, st1)),
        )

    def rewrite(self, op, x, **_):
        return _replace_slice_partition_with_split(self, op, x)


class ReplaceSlicePartitionWithSplit3(RewriteRuleClassBase):
    """Replace three constant sibling Slices covering an axis with Split."""

    def pattern(
        self,
        op,
        x,
        s0,
        e0,
        a0,
        st0,
        s1,
        e1,
        a1,
        st1,
        s2,
        e2,
        a2,
        st2,
    ):
        return (
            op.Slice(x, s0, e0, a0, st0),
            op.Slice(x, s1, e1, a1, st1),
            op.Slice(x, s2, e2, a2, st2),
        )

    def check(
        self,
        context,
        x,
        s0,
        e0,
        a0,
        st0,
        s1,
        e1,
        a1,
        st1,
        s2,
        e2,
        a2,
        st2,
    ):
        return _check_complete_slice_partition(
            self,
            context,
            x,
            (
                (s0, e0, a0, st0),
                (s1, e1, a1, st1),
                (s2, e2, a2, st2),
            ),
        )

    def rewrite(self, op, x, **_):
        return _replace_slice_partition_with_split(self, op, x)


class ReplaceSlicePartitionWithSplit4(RewriteRuleClassBase):
    """Replace four constant sibling Slices covering an axis with Split."""

    def pattern(
        self,
        op,
        x,
        s0,
        e0,
        a0,
        st0,
        s1,
        e1,
        a1,
        st1,
        s2,
        e2,
        a2,
        st2,
        s3,
        e3,
        a3,
        st3,
    ):
        return (
            op.Slice(x, s0, e0, a0, st0),
            op.Slice(x, s1, e1, a1, st1),
            op.Slice(x, s2, e2, a2, st2),
            op.Slice(x, s3, e3, a3, st3),
        )

    def check(
        self,
        context,
        x,
        s0,
        e0,
        a0,
        st0,
        s1,
        e1,
        a1,
        st1,
        s2,
        e2,
        a2,
        st2,
        s3,
        e3,
        a3,
        st3,
    ):
        return _check_complete_slice_partition(
            self,
            context,
            x,
            (
                (s0, e0, a0, st0),
                (s1, e1, a1, st1),
                (s2, e2, a2, st2),
                (s3, e3, a3, st3),
            ),
        )

    def rewrite(self, op, x, **_):
        return _replace_slice_partition_with_split(self, op, x)


class _ReplaceGatherSlicePartitionWithSplit2Base(RewriteRuleClassBase):
    def __init__(self, gather_first: bool):
        super().__init__(
            "ReplaceGatherSlicePartitionWithSplit2GatherFirst"
            if gather_first
            else "ReplaceGatherSlicePartitionWithSplit2SliceFirst"
        )
        self._gather_first = gather_first

    def check(
        self,
        context,
        x,
        index,
        starts,
        ends,
        axes,
        steps,
    ) -> MatchResult:
        result = MatchResult()
        opset = _default_opset(context)
        if opset is not None and opset < 13:
            return result.fail(
                "Gather/Slice partition requires input-form Squeeze (opset >= 13)."
            )
        index_array = _ints(index, ranks=(0,))
        slice_range = _read_slice_range(x, starts, ends, axes, steps)
        if index_array is None or slice_range is None or x.shape is None:
            return result.fail("Gather/Slice parameters must be constant and valid.")
        gather_output = context.output_values[0 if self._gather_first else 1]
        gather_node = gather_output.producer()
        if gather_node is None:
            return result.fail("Matched Gather output has no producer.")
        axis = _node_axis(gather_node, x.shape.rank())
        if axis is None or axis != slice_range[0]:
            return result.fail("Gather and Slice axes differ.")
        dim = x.shape[axis]
        assert isinstance(dim, int)
        selected = int(index_array.item())
        selected = selected + dim if selected < 0 else selected
        if not 0 <= selected < dim:
            return result.fail("Gather index is out of range.")
        ranges = [(selected, selected + 1), (slice_range[1], slice_range[2])]
        ordered = sorted(enumerate(ranges), key=lambda item: item[1][0])
        if (
            ordered[0][1][0] != 0
            or ordered[0][1][1] != ordered[1][1][0]
            or ordered[1][1][1] != dim
        ):
            return result.fail("Gather and Slice do not partition the complete axis.")
        self._sizes = [ordered[0][1][1], dim - ordered[0][1][1]]
        self._gather_split_slot = next(
            slot for slot, (kind, _) in enumerate(ordered) if kind == 0
        )
        self._slice_split_slot = 1 - self._gather_split_slot
        self._axis = axis
        return result

    def rewrite(self, op, x, **_):
        split_outputs = op.Split(x, _const(op, self._sizes), axis=self._axis, _outputs=2)
        gather_output = op.Squeeze(
            split_outputs[self._gather_split_slot], _const(op, [self._axis])
        )
        slice_output = split_outputs[self._slice_split_slot]
        return (
            (gather_output, slice_output)
            if self._gather_first
            else (slice_output, gather_output)
        )


class ReplaceGatherSlicePartitionWithSplit2GatherFirst(
    _ReplaceGatherSlicePartitionWithSplit2Base
):
    def __init__(self):
        super().__init__(True)

    def pattern(self, op, x, index, starts, ends, axes, steps):
        return (
            op.Gather(x, index),
            op.Slice(x, starts, ends, axes, steps),
        )


class ReplaceGatherSlicePartitionWithSplit2SliceFirst(
    _ReplaceGatherSlicePartitionWithSplit2Base
):
    def __init__(self):
        super().__init__(False)

    def pattern(self, op, x, index, starts, ends, axes, steps):
        return (
            op.Slice(x, starts, ends, axes, steps),
            op.Gather(x, index),
        )


class _ReplaceGatherSlicePartitionWithSplit3Base(RewriteRuleClassBase):
    def __init__(self, name: str, gather_count: int):
        super().__init__(name)
        self._gather_count = gather_count

    def _check(self, context, x, gather_indices, slices) -> MatchResult:
        result = MatchResult()
        opset = _default_opset(context)
        if opset is not None and opset < 13:
            return result.fail(
                "Gather/Slice partition requires input-form Squeeze (opset >= 13)."
            )
        if x.shape is None:
            return result.fail("Partition input shape must be known.")
        ranges: list[tuple[int, int, bool]] = []
        common_axis = None
        gather_nodes = [
            value.producer() for value in context.output_values[: self._gather_count]
        ]
        if any(node is None for node in gather_nodes):
            return result.fail("Matched Gather output has no producer.")
        for index, gather_node in zip(gather_indices, gather_nodes):
            index_array = _ints(index, ranks=(0,))
            assert gather_node is not None
            axis = _node_axis(gather_node, x.shape.rank())
            if index_array is None or axis is None or not isinstance(x.shape[axis], int):
                return result.fail("Gather parameters and axis dimension must be static.")
            dim = x.shape[axis]
            selected = int(index_array.item())
            selected = selected + dim if selected < 0 else selected
            if not 0 <= selected < dim:
                return result.fail("Gather index is out of range.")
            common_axis = axis if common_axis is None else common_axis
            if axis != common_axis:
                return result.fail("Partition operators use different axes.")
            ranges.append((selected, selected + 1, True))
        for starts, ends, axes, steps in slices:
            item = _read_slice_range(x, starts, ends, axes, steps)
            if item is None:
                return result.fail("Slice parameters must be constant and valid.")
            axis, begin, finish = item
            common_axis = axis if common_axis is None else common_axis
            if axis != common_axis:
                return result.fail("Partition operators use different axes.")
            ranges.append((begin, finish, False))
        assert common_axis is not None
        dim = x.shape[common_axis]
        assert isinstance(dim, int)
        ordered = sorted(enumerate(ranges), key=lambda item: item[1][0])
        expected = 0
        for _, (begin, finish, _) in ordered:
            if begin != expected:
                return result.fail("Ranges are not a contiguous complete partition.")
            expected = finish
        if expected != dim:
            return result.fail("Ranges do not cover the complete axis.")
        self._sizes = [finish - begin for _, (begin, finish, _) in ordered]
        self._split_slots = [0] * len(ranges)
        for split_slot, (original_slot, _) in enumerate(ordered):
            self._split_slots[original_slot] = split_slot
        self._axis = common_axis
        return result

    def rewrite(self, op, x, **_):
        splits = op.Split(
            x,
            _const(op, self._sizes),
            axis=self._axis,
            _outputs=len(self._sizes),
        )
        axes = _const(op, [self._axis])
        outputs = []
        for original_slot, split_slot in enumerate(self._split_slots):
            value = splits[split_slot]
            outputs.append(
                op.Squeeze(value, axes) if original_slot < self._gather_count else value
            )
        return tuple(outputs)


class ReplaceOneGatherTwoSlicesPartitionWithSplit(_ReplaceGatherSlicePartitionWithSplit3Base):
    def __init__(self):
        super().__init__("ReplaceOneGatherTwoSlicesPartitionWithSplit", 1)

    def pattern(
        self,
        op,
        x,
        index,
        s0,
        e0,
        a0,
        st0,
        e1,
        a1,
        st1,
    ):
        return (
            op.Gather(x, index),
            op.Slice(x, s0, e0, a0, st0),
            op.Slice(x, e0, e1, a1, st1),
        )

    def check(
        self,
        context,
        x,
        index,
        s0,
        e0,
        a0,
        st0,
        e1,
        a1,
        st1,
    ):
        return self._check(
            context,
            x,
            (index,),
            ((s0, e0, a0, st0), (e0, e1, a1, st1)),
        )


class ReplaceTwoGathersOneSlicePartitionWithSplit(_ReplaceGatherSlicePartitionWithSplit3Base):
    def __init__(self):
        super().__init__("ReplaceTwoGathersOneSlicePartitionWithSplit", 2)

    def pattern(
        self,
        op,
        x,
        i0,
        i1,
        starts,
        ends,
        axes,
        steps,
    ):
        return (
            op.Gather(x, i0),
            op.Gather(x, i1),
            op.Slice(x, starts, ends, axes, steps),
        )

    def check(
        self,
        context,
        x,
        i0,
        i1,
        starts,
        ends,
        axes,
        steps,
    ):
        return self._check(
            context,
            x,
            (i0, i1),
            ((starts, ends, axes, steps),),
        )


class _EliminateSplitConcatRoundTripBase(RewriteRuleClassBase):
    def __init__(self, count: int):
        super().__init__(f"EliminateSplitConcatRoundTrip{count}", remove_nodes=False)

    def check(self, context, x, split, concat_axis) -> MatchResult:
        del split
        result = MatchResult()
        rank = x.shape.rank() if x.shape is not None else None
        split_node = next(node for node in context.nodes if node.op_type == "Split")
        split = _node_axis(split_node, rank)
        concat = _axis(concat_axis, rank)
        if split is None or concat is None or split != concat:
            return result.fail("Split and Concat axes differ or cannot be normalized.")
        graph_outputs = set(context.graph_or_function.outputs)
        if any(
            len(output.uses()) != 1 or output in graph_outputs for output in split_node.outputs
        ):
            return result.fail("Every Split output must be used only by the matched Concat.")
        return result

    def rewrite(self, op, x, **_):
        return op.Identity(x)


class EliminateSplitConcatRoundTrip2(_EliminateSplitConcatRoundTripBase):
    def __init__(self):
        super().__init__(2)

    def pattern(self, op, x, split, concat_axis):
        y0, y1 = op.Split(x, split, _outputs=2)
        return op.Concat(y0, y1, axis=concat_axis)


class EliminateSplitConcatRoundTrip3(_EliminateSplitConcatRoundTripBase):
    def __init__(self):
        super().__init__(3)

    def pattern(self, op, x, split, concat_axis):
        y0, y1, y2 = op.Split(x, split, _outputs=3)
        return op.Concat(y0, y1, y2, axis=concat_axis)


class EliminateSplitConcatRoundTrip4(_EliminateSplitConcatRoundTripBase):
    def __init__(self):
        super().__init__(4)

    def pattern(self, op, x, split, concat_axis):
        y0, y1, y2, y3 = op.Split(x, split, _outputs=4)
        return op.Concat(y0, y1, y2, y3, axis=concat_axis)


def _sequence_index_selects(expected: int, count: int):
    def check(context, node) -> bool:
        del context
        index = _ints(node.inputs[1], ranks=(0,))
        if index is None:
            return False
        value = int(index.item())
        if value < 0:
            value += count
        return value == expected

    return check


class _EliminateSequenceConstructAtPermutationBase(RewriteRuleClassBase):
    def __init__(self, count: int):
        super().__init__(f"EliminateSequenceConstructAtPermutation{count}", remove_nodes=False)
        self._count = count

    def check(self, context, **bindings) -> MatchResult:
        result = MatchResult()
        sequence_node = next(
            node for node in context.nodes if node.op_type == "SequenceConstruct"
        )
        sequence = sequence_node.outputs[0]
        if len(sequence.uses()) != self._count:
            return result.fail("The constructed sequence has unmatched users.")
        seen: set[int] = set()
        for i in range(self._count):
            index = _ints(bindings[f"i{i}"], ranks=(0,))
            if index is None:
                return result.fail("SequenceAt index must be a constant integer scalar.")
            value = int(index.item())
            if value < 0:
                value += self._count
            if value < 0 or value >= self._count or value in seen:
                return result.fail("SequenceAt indices must select every item exactly once.")
            seen.add(value)
        self._order = [
            (int(_ints(bindings[f"i{i}"], ranks=(0,)).item()) + self._count) % self._count
            for i in range(self._count)
        ]
        return result

    def rewrite(self, op, **bindings):
        inputs = [bindings[f"x{i}"] for i in range(self._count)]
        return tuple(op.Identity(inputs[index]) for index in self._order)


class EliminateSequenceConstructAtPermutation2(_EliminateSequenceConstructAtPermutationBase):
    def __init__(self):
        super().__init__(2)

    def pattern(self, op, x0, x1, i0, i1):
        sequence = op.SequenceConstruct(x0, x1)
        return (
            op.SequenceAt(sequence, i0, _check=_sequence_index_selects(0, 2)),
            op.SequenceAt(sequence, i1, _check=_sequence_index_selects(1, 2)),
        )


class EliminateSequenceConstructAtPermutation3(_EliminateSequenceConstructAtPermutationBase):
    def __init__(self):
        super().__init__(3)

    def pattern(self, op, x0, x1, x2, i0, i1, i2):
        sequence = op.SequenceConstruct(x0, x1, x2)
        return (
            op.SequenceAt(sequence, i0, _check=_sequence_index_selects(0, 3)),
            op.SequenceAt(sequence, i1, _check=_sequence_index_selects(1, 3)),
            op.SequenceAt(sequence, i2, _check=_sequence_index_selects(2, 3)),
        )


class EliminateSequenceConstructAtPermutation4(_EliminateSequenceConstructAtPermutationBase):
    def __init__(self):
        super().__init__(4)

    def pattern(self, op, x0, x1, x2, x3, i0, i1, i2, i3):
        sequence = op.SequenceConstruct(x0, x1, x2, x3)
        return (
            op.SequenceAt(sequence, i0, _check=_sequence_index_selects(0, 4)),
            op.SequenceAt(sequence, i1, _check=_sequence_index_selects(1, 4)),
            op.SequenceAt(sequence, i2, _check=_sequence_index_selects(2, 4)),
            op.SequenceAt(sequence, i3, _check=_sequence_index_selects(3, 4)),
        )


_move_before_duplicate_concat_rules = [
    MoveUnaryBeforeDuplicateConcat.rule(op_type) for op_type in _ELEMENTWISE_UNARY_OPS
]
_move_before_duplicate_concat_rules.extend(
    MoveScalarBinaryBeforeDuplicateConcat.rule(op_type)
    for op_type in ("Add", "Div", "Mul", "Sub")
)

_move_before_gather_rules = [
    MoveUnaryBeforeGather.rule(op_type) for op_type in _ELEMENTWISE_UNARY_OPS
]
_move_before_gather_rules.extend(
    MoveBroadcastBinaryBeforeGather.rule(op_type) for op_type in ("Add", "Div", "Mul", "Sub")
)

rules = RewriteRuleSet(
    [
        RemoveEmptyConcatInputs.rule(),
        *_move_before_duplicate_concat_rules,
        MoveCastBeforeDuplicateConcat.rule(),
        ComposeConsecutiveGathers.rule(),
        SelectConcatInputWithGather.rule(),
        *_move_before_gather_rules,
        MoveCastBeforeGather.rule(),
        ReplaceGatherPartitionWithSplit2.rule(),
        ReplaceGatherPartitionWithSplit3.rule(),
        ReplaceGatherPartitionWithSplit4.rule(),
        ReplaceSlicePartitionWithSplit2.rule(),
        ReplaceSlicePartitionWithSplit3.rule(),
        ReplaceSlicePartitionWithSplit4.rule(),
        ReplaceGatherSlicePartitionWithSplit2GatherFirst.rule(),
        ReplaceGatherSlicePartitionWithSplit2SliceFirst.rule(),
        ReplaceOneGatherTwoSlicesPartitionWithSplit.rule(),
        ReplaceTwoGathersOneSlicePartitionWithSplit.rule(),
        ReplaceArithmeticProgressionGatherWithSlice.rule(),
        SelectConcatInputWithSlice.rule(),
        FuseConsecutiveSlices.rule(),
        FuseSpatialSlicesIntoSpaceToDepth.rule(),
        EliminateSplitConcatRoundTrip2.rule(),
        EliminateSplitConcatRoundTrip3.rule(),
        EliminateSplitConcatRoundTrip4.rule(),
        EliminateSequenceConstructAtPermutation2.rule(),
        EliminateSequenceConstructAtPermutation3.rule(),
        EliminateSequenceConstructAtPermutation4.rule(),
    ]
)
