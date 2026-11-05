# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Sound canonicalization rules that are not covered by the basic rule set."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleClassBase, RewriteRuleSet


def _constant_array(value: ir.Value | None) -> np.ndarray | None:
    if value is None:
        return None
    array = _ir_utils.get_numpy_value(value)
    return None if array is None else np.asarray(array)


def _default_opset(context) -> int | None:
    local_imports = context.graph_or_function.opset_imports
    if "" in local_imports:
        return local_imports[""]
    if "ai.onnx" in local_imports:
        return local_imports["ai.onnx"]
    return context.model.opset_imports.get("", context.model.opset_imports.get("ai.onnx"))


def _attr_is_ref(node: ir.Node, name: str) -> bool:
    attribute = node.attributes.get(name)
    return attribute is not None and attribute.is_ref()


def _same_constant(left: ir.Value | None, right: ir.Value | None) -> bool:
    if left is None or right is None:
        return left is right
    left_array = _constant_array(left)
    right_array = _constant_array(right)
    if left_array is None or right_array is None:
        return False
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in "OSU":
        return np.array_equal(left_array, right_array)
    return left_array.tobytes() == right_array.tobytes()


def _is_default_pad_value(value: ir.Value) -> bool:
    array = _constant_array(value)
    if array is None or array.size != 1 or array.item() != 0:
        return False
    return not bool(np.signbit(array.item()))


def _normalize_pad(
    pads_value: ir.Value,
    axes_value: ir.Value | None,
    rank: int,
) -> np.ndarray | None:
    pads = _constant_array(pads_value)
    if pads is None or pads.ndim != 1 or pads.size % 2:
        return None
    if not np.issubdtype(pads.dtype, np.integer):
        return None
    pads = pads.astype(np.int64)
    if np.any(pads < 0):
        return None
    if axes_value is None:
        return pads if pads.size == 2 * rank else None

    axes = _constant_array(axes_value)
    if axes is None or axes.ndim != 1 or axes.size * 2 != pads.size:
        return None
    if not np.issubdtype(axes.dtype, np.integer):
        return None
    axes = axes.astype(np.int64)
    normalized = np.zeros(2 * rank, dtype=np.int64)
    seen: set[int] = set()
    for index, raw_axis in enumerate(axes):
        axis = int(raw_axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank or axis in seen:
            return None
        seen.add(axis)
        normalized[axis] = pads[index]
        normalized[rank + axis] = pads[axes.size + index]
    return normalized


class PadPadFusion(RewriteRuleClassBase):
    """Fuses two non-negative constant-mode Pads with equal fill values."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, x):
        first = op.Pad(
            x,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["first"],
        )
        return op.Pad(
            first,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["second"],
        )

    def check(self, context, x: ir.Value, first: ir.Value, second: ir.Value) -> MatchResult:
        del context
        result = MatchResult()
        first_node = first.producer()
        second_node = second.producer()
        if first_node is None or second_node is None or x.shape is None:
            return result.fail("Pad inputs and rank must be known.")
        if _attr_is_ref(first_node, "mode") or _attr_is_ref(second_node, "mode"):
            return result.fail("Pad mode reference attributes are unsupported.")
        if (
            first_node.attributes.get_string("mode", "constant") != "constant"
            or second_node.attributes.get_string("mode", "constant") != "constant"
        ):
            return result.fail("Only constant-mode Pads can be combined.")
        if first.is_graph_output() or len(list(first.uses())) != 1:
            return result.fail("The first Pad must be removable to avoid duplicating padding.")
        if len(first_node.inputs) < 2 or len(second_node.inputs) < 2:
            return result.fail("Pads must use the input-form pads parameter.")

        rank = x.shape.rank()
        first_axes = first_node.inputs[3] if len(first_node.inputs) > 3 else None
        second_axes = second_node.inputs[3] if len(second_node.inputs) > 3 else None
        first_pads = _normalize_pad(first_node.inputs[1], first_axes, rank)
        second_pads = _normalize_pad(second_node.inputs[1], second_axes, rank)
        if first_pads is None or second_pads is None:
            return result.fail("Pads and axes must be valid non-negative constants.")

        first_value = first_node.inputs[2] if len(first_node.inputs) > 2 else None
        second_value = second_node.inputs[2] if len(second_node.inputs) > 2 else None
        if first_value is None and second_value is not None:
            if not _is_default_pad_value(second_value):
                return result.fail("Pad fill values differ.")
        elif second_value is None and first_value is not None:
            if not _is_default_pad_value(first_value):
                return result.fail("Pad fill values differ.")
        elif not _same_constant(first_value, second_value):
            return result.fail("Pad fill values differ.")

        summed = first_pads.astype(object) + second_pads.astype(object)
        if any(value > np.iinfo(np.int64).max for value in summed):
            return result.fail("Combined pads overflow int64.")
        self._summed_pads = np.asarray(summed, dtype=np.int64)
        self._constant_value = second_value if second_value is not None else first_value
        return result

    def rewrite(self, op, x: ir.Value, first: ir.Value, second: ir.Value):
        del first
        pads = op.initializer(
            ir.tensor(self._summed_pads, dtype=ir.DataType.INT64),
            name=f"{second.name}_pads",
        )
        if self._constant_value is None:
            return op.Pad(x, pads, mode="constant")
        return op.Pad(x, pads, self._constant_value, mode="constant")


class ReshapeIdentity(RewriteRuleClassBase):
    """Removes Reshape(x, [0, ...]) when it copies every input dimension."""

    def pattern(self, op, x, shape):
        return op.Reshape(
            x,
            shape,
            _allow_other_attributes=True,
            _outputs=["output"],
        )

    def check(self, context, x: ir.Value, shape: ir.Value, output: ir.Value) -> MatchResult:
        del context
        result = MatchResult()
        node = output.producer()
        if (
            node is None
            or _attr_is_ref(node, "allowzero")
            or node.attributes.get_int("allowzero", 0) != 0
        ):
            return result.fail("Reshape allowzero must be disabled.")
        shape_array = _constant_array(shape)
        if (
            shape_array is None
            or shape_array.ndim != 1
            or not np.issubdtype(shape_array.dtype, np.integer)
            or np.any(shape_array != 0)
        ):
            return result.fail("Reshape shape must be an all-zero integer vector.")
        if x.shape is None or x.shape.rank() != shape_array.size:
            return result.fail("Input rank must equal the Reshape shape length.")
        return result

    def rewrite(self, op, x: ir.Value, **_):
        return op.Identity(x)


class _ReduceArgTopKBase(RewriteRuleClassBase):
    reduce_op: str
    arg_op: str
    largest: int

    def check(
        self,
        context,
        x: ir.Value,
        axes: ir.Value,
        values: ir.Value,
        indices: ir.Value,
    ) -> MatchResult:
        result = MatchResult()
        if (opset := _default_opset(context)) is None or opset < 18:
            return result.fail("The axes-input reduction form requires ONNX opset 18.")
        supported_types = {
            ir.DataType.INT8,
            ir.DataType.INT32,
            ir.DataType.INT64,
            ir.DataType.UINT8,
            ir.DataType.UINT32,
            ir.DataType.UINT64,
        }
        if x.dtype not in supported_types:
            return result.fail(
                "Only integer reductions are exactly interchangeable with TopK."
            )
        if x.shape is None:
            return result.fail("Input rank is needed to compare reduction axes.")
        axes_array = _constant_array(axes)
        if (
            axes_array is None
            or axes_array.ndim != 1
            or axes_array.size != 1
            or not np.issubdtype(axes_array.dtype, np.integer)
        ):
            return result.fail("Reduce axes must be one constant integer.")

        reduce_node = values.producer()
        arg_node = indices.producer()
        if reduce_node is None or arg_node is None:
            return result.fail("Reduction producers are unavailable.")
        if any(
            _attr_is_ref(node, name)
            for node, names in (
                (reduce_node, ("noop_with_empty_axes", "keepdims")),
                (arg_node, ("select_last_index", "keepdims", "axis")),
            )
            for name in names
        ):
            return result.fail("Reduction reference attributes are unsupported.")
        if reduce_node.attributes.get_int("noop_with_empty_axes", 0) != 0:
            return result.fail("noop_with_empty_axes is unsupported.")
        if arg_node.attributes.get_int("select_last_index", 0) != 0:
            return result.fail("TopK selects the lowest index on ties.")
        reduce_keepdims = reduce_node.attributes.get_int("keepdims", 1)
        arg_keepdims = arg_node.attributes.get_int("keepdims", 1)
        if reduce_keepdims != arg_keepdims:
            return result.fail("Reduce and Arg keepdims attributes differ.")

        rank = x.shape.rank()
        reduce_axis = int(axes_array.item())
        arg_axis = arg_node.attributes.get_int("axis", 0)
        if reduce_axis < 0:
            reduce_axis += rank
        if arg_axis < 0:
            arg_axis += rank
        if (
            reduce_axis < 0
            or reduce_axis >= rank
            or arg_axis < 0
            or arg_axis >= rank
            or reduce_axis != arg_axis
        ):
            return result.fail("Reduce and Arg axes differ or are out of range.")
        self._axis = reduce_axis
        self._keepdims = reduce_keepdims
        return result

    def rewrite(self, op, x: ir.Value, **_):
        k = op.initializer(ir.tensor([1], dtype=ir.DataType.INT64), name=f"{x.name}_topk_k")
        top_values, top_indices = op.TopK(
            x,
            k,
            axis=self._axis,
            largest=self.largest,
            sorted=1,
            _outputs=2,
        )
        if self._keepdims:
            return top_values, top_indices
        axes = op.initializer(
            ir.tensor([self._axis], dtype=ir.DataType.INT64),
            name=f"{x.name}_topk_axes",
        )
        return op.Squeeze(top_values, axes), op.Squeeze(top_indices, axes)


class ReduceMaxArgMaxTopK(_ReduceArgTopKBase):
    """Combines exact integer ReduceMax and ArgMax siblings into TopK."""

    reduce_op = "ReduceMax"
    arg_op = "ArgMax"
    largest = 1

    def pattern(self, op, x, axes):
        values = op.ReduceMax(
            x,
            axes,
            _allow_other_attributes=True,
            _outputs=["values"],
        )
        indices = op.ArgMax(x, _allow_other_attributes=True, _outputs=["indices"])
        return values, indices


class ReduceMinArgMinTopK(_ReduceArgTopKBase):
    """Combines exact integer ReduceMin and ArgMin siblings into TopK."""

    reduce_op = "ReduceMin"
    arg_op = "ArgMin"
    largest = 0

    def pattern(self, op, x, axes):
        values = op.ReduceMin(
            x,
            axes,
            _allow_other_attributes=True,
            _outputs=["values"],
        )
        indices = op.ArgMin(x, _allow_other_attributes=True, _outputs=["indices"])
        return values, indices


def _neutral_vector_check(
    context,
    x: ir.Value,
    constant: ir.Value,
    output: ir.Value,
    *,
    neutral: int,
    float_safe: bool,
) -> MatchResult:
    del context
    result = MatchResult()
    values = _constant_array(constant)
    if values is None or values.ndim != 1 or values.size == 0 or np.any(values != neutral):
        return result.fail("The constant is not a non-empty neutral vector.")
    if (
        x.shape is None
        or output.shape is None
        or not _ir_utils.same_shape(x.shape, output.shape)
    ):
        return result.fail("Removing the operation could change its broadcasted shape.")
    if x.shape.rank() == 0 or x.shape.is_dynamic(-1) or x.shape[-1] != values.size:
        return result.fail("The neutral vector does not match the final input dimension.")
    if not float_safe and (x.dtype is None or x.dtype.is_floating_point()):
        return result.fail("Floating addition with zero does not preserve signed zero.")
    return result


def _identity(op, x, **_):
    return op.Identity(x)


def _add_vector_right(op, x, constant):
    return op.Add(x, constant, _outputs=["output"])


def _add_vector_left(op, x, constant):
    return op.Add(constant, x, _outputs=["output"])


def _sub_vector_right(op, x, constant):
    return op.Sub(x, constant, _outputs=["output"])


def _mul_vector_right(op, x, constant):
    return op.Mul(x, constant, _outputs=["output"])


def _mul_vector_left(op, x, constant):
    return op.Mul(constant, x, _outputs=["output"])


def _div_vector_right(op, x, constant):
    return op.Div(x, constant, _outputs=["output"])


def _zero_vector_check(context, x, constant, output):
    return _neutral_vector_check(context, x, constant, output, neutral=0, float_safe=False)


def _one_vector_check(context, x, constant, output):
    return _neutral_vector_check(context, x, constant, output, neutral=1, float_safe=True)


class CastTranspose(RewriteRuleClassBase):
    """Moves an elementwise Cast before a Transpose without changing values."""

    def pattern(self, op, x):
        transposed = op.Transpose(x, _allow_other_attributes=True, _outputs=["transposed"])
        return op.Cast(transposed, _allow_other_attributes=True, _outputs=["cast_output"])

    def check(
        self,
        context,
        x: ir.Value,
        transposed: ir.Value,
        cast_output: ir.Value,
    ) -> MatchResult:
        del context, x, cast_output
        result = MatchResult()
        if transposed.is_graph_output() or len(list(transposed.uses())) != 1:
            return result.fail("Transpose must be removable to avoid duplicating layout work.")
        return result

    def rewrite(
        self,
        op,
        x: ir.Value,
        transposed: ir.Value,
        cast_output: ir.Value,
    ):
        transpose_node = transposed.producer()
        cast_node = cast_output.producer()
        assert transpose_node is not None and cast_node is not None
        casted = op.op("Cast", x, **cast_node.attributes)
        return op.op("Transpose", casted, **transpose_node.attributes)


class _CastLayoutBase(RewriteRuleClassBase):
    def rewrite(
        self,
        op,
        x: ir.Value,
        layout_output: ir.Value,
        cast_output: ir.Value,
        **_,
    ):
        layout_node = layout_output.producer()
        cast_node = cast_output.producer()
        assert layout_node is not None and cast_node is not None
        casted = op.op("Cast", x, **cast_node.attributes)
        return op.op(
            layout_node.op_type,
            casted,
            *layout_node.inputs[1:],
            **layout_node.attributes,
        )


class CastReshape(_CastLayoutBase):
    """Moves an elementwise Cast before a Reshape."""

    def pattern(self, op, x, shape):
        layout_output = op.Reshape(
            x,
            shape,
            _allow_other_attributes=True,
            _outputs=["layout_output"],
        )
        return op.Cast(layout_output, _allow_other_attributes=True, _outputs=["cast_output"])


class CastSqueeze(_CastLayoutBase):
    """Moves an elementwise Cast before a Squeeze."""

    def pattern(self, op, x):
        layout_output = op.Squeeze(
            x,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["layout_output"],
        )
        return op.Cast(layout_output, _allow_other_attributes=True, _outputs=["cast_output"])


class CastUnsqueeze(_CastLayoutBase):
    """Moves an elementwise Cast before an Unsqueeze."""

    def pattern(self, op, x):
        layout_output = op.Unsqueeze(
            x,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["layout_output"],
        )
        return op.Cast(layout_output, _allow_other_attributes=True, _outputs=["cast_output"])


class SwapRangeAddScalar(RewriteRuleClassBase):
    """Moves a checked constant integer offset into Range start and limit."""

    def __init__(self):
        super().__init__(remove_nodes=False)

    def pattern(self, op, start, limit, delta, offset):
        range_output = op.Range(start, limit, delta, _outputs=["range_output"])
        return op.Add(range_output, offset)

    def check(
        self,
        context,
        start: ir.Value,
        limit: ir.Value,
        delta: ir.Value,
        offset: ir.Value,
        range_output: ir.Value,
    ) -> MatchResult:
        result = MatchResult()
        if (opset := _default_opset(context)) is None or opset < 11:
            return result.fail("Range requires ONNX opset 11 or newer.")
        constants = [_constant_array(value) for value in (start, limit, delta, offset)]
        if any(value is None or value.size != 1 for value in constants):
            return result.fail("Range parameters and offset must be scalar constants.")
        assert all(value is not None for value in constants)
        dtypes = {value.dtype for value in constants}
        if len(dtypes) != 1:
            return result.fail("Range parameters and offset must have the same dtype.")
        dtype = constants[0].dtype
        if dtype not in (np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64)):
            return result.fail("Only ONNX integer Range dtypes are supported.")
        start_value, limit_value, delta_value, offset_value = (
            int(value.item()) for value in constants
        )
        if delta_value == 0:
            return result.fail("Range delta must be nonzero.")
        bounds = np.iinfo(dtype)
        shifted_start = start_value + offset_value
        shifted_limit = limit_value + offset_value
        if not (
            bounds.min <= shifted_start <= bounds.max
            and bounds.min <= shifted_limit <= bounds.max
        ):
            return result.fail("Shifted Range bounds overflow the input dtype.")
        if range_output.is_graph_output() or len(list(range_output.uses())) != 1:
            return result.fail("The Range output is shared, so the rewrite removes no work.")
        self._dtype = ir.DataType.from_numpy(dtype)
        self._shifted_start = shifted_start
        self._shifted_limit = shifted_limit
        return result

    def rewrite(
        self,
        op,
        start: ir.Value,
        limit: ir.Value,
        delta: ir.Value,
        offset: ir.Value,
        **_,
    ):
        del start, limit, offset
        shifted_start = op.initializer(
            ir.tensor(self._shifted_start, dtype=self._dtype), name="range_shifted_start"
        )
        shifted_limit = op.initializer(
            ir.tensor(self._shifted_limit, dtype=self._dtype), name="range_shifted_limit"
        )
        return op.Range(shifted_start, shifted_limit, delta)


pad_pad_fusion_rule = PadPadFusion.rule()
reshape_identity_rule = ReshapeIdentity.rule()
reduce_max_argmax_topk_rule = ReduceMaxArgMaxTopK.rule(remove_nodes=False)
reduce_min_argmin_topk_rule = ReduceMinArgMinTopK.rule(remove_nodes=False)
add_vector_right_rule = RewriteRule(_add_vector_right, _identity, _zero_vector_check)
add_vector_left_rule = RewriteRule(_add_vector_left, _identity, _zero_vector_check)
sub_vector_right_rule = RewriteRule(_sub_vector_right, _identity, _zero_vector_check)
mul_vector_right_rule = RewriteRule(_mul_vector_right, _identity, _one_vector_check)
mul_vector_left_rule = RewriteRule(_mul_vector_left, _identity, _one_vector_check)
div_vector_right_rule = RewriteRule(_div_vector_right, _identity, _one_vector_check)
cast_transpose_rule = CastTranspose.rule(remove_nodes=False)
cast_reshape_rule = CastReshape.rule(remove_nodes=False)
cast_squeeze_rule = CastSqueeze.rule(remove_nodes=False)
cast_unsqueeze_rule = CastUnsqueeze.rule(remove_nodes=False)
swap_range_add_scalar_rule = SwapRangeAddScalar.rule()

rules = RewriteRuleSet(
    [
        pad_pad_fusion_rule,
        reshape_identity_rule,
        reduce_max_argmax_topk_rule,
        reduce_min_argmin_topk_rule,
        add_vector_right_rule,
        add_vector_left_rule,
        sub_vector_right_rule,
        mul_vector_right_rule,
        mul_vector_left_rule,
        div_vector_right_rule,
        cast_transpose_rule,
        cast_reshape_rule,
        cast_squeeze_rule,
        cast_unsqueeze_rule,
        swap_range_add_scalar_rule,
    ]
)
