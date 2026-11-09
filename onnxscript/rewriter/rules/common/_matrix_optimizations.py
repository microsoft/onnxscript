# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""General matrix rewrites that preserve ONNX operator semantics."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet
from onnxscript.rewriter.rules.common import _broadcast_to_matmul

_GEMM_TYPES = frozenset(
    {
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
    }
)
_EXACT_UNIT_REDUCTION_TYPES = frozenset(
    {
        ir.DataType.UINT32,
        ir.DataType.UINT64,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }
)


def _main_opset(context) -> int | None:
    graph_opsets = context.graph_or_function.opset_imports
    return graph_opsets.get(
        "",
        graph_opsets.get(
            "ai.onnx",
            context.model.opset_imports.get("", context.model.opset_imports.get("ai.onnx")),
        ),
    )


def _has_reference_attribute(node: ir.Node, names: set[str] | None = None) -> bool:
    return any(
        (names is None or name in names) and attribute.is_ref()
        for name, attribute in node.attributes.items()
    )


def _is_private_intermediate(context, value: ir.Value) -> bool:
    return len(value.uses()) == 1 and value not in set(context.graph_or_function.outputs)


def _matrix_transpose(node: ir.Node, value: ir.Value) -> bool:
    if _has_reference_attribute(node, {"perm"}):
        return False
    perm = node.attributes.get_ints("perm")
    if perm:
        return tuple(perm) == (1, 0)
    return _ir_utils.has_rank(value, 2)


def _node_attributes(value: ir.Value) -> dict[str, object]:
    node = value.producer()
    assert node is not None
    return dict(node.attributes)


class _TransposeMatMulToGemm(RewriteRuleClassBase):
    position: ClassVar[int] = 0

    def __init__(self) -> None:
        input_name = "A" if self.position == 0 else "B"
        super().__init__(f"FoldTranspose{input_name}MatMulIntoGemm", remove_nodes=False)

    def pattern(self, op, x, y):
        if self.position == 0:
            return op.MatMul(op.Transpose(x, _outputs=["transposed"]), y)
        return op.MatMul(x, op.Transpose(y, _outputs=["transposed"]))

    def check(self, context, x, y, transposed, **_) -> MatchResult:
        result = MatchResult()
        if (_main_opset(context) or 0) < 11:
            return result.fail("Two-input Gemm requires ONNX opset 11 or newer.")
        if not (_ir_utils.has_rank(x, 2) and _ir_utils.has_rank(y, 2)):
            return result.fail("Gemm inputs must both have rank two.")
        if x.dtype not in _GEMM_TYPES or y.dtype != x.dtype:
            return result.fail("MatMul input types are not supported by Gemm.")
        if x.dtype == ir.DataType.BFLOAT16 and (_main_opset(context) or 0) < 13:
            return result.fail("Gemm supports BFLOAT16 starting in ONNX opset 13.")
        transpose = transposed.producer()
        assert transpose is not None
        if not _matrix_transpose(transpose, x if self.position == 0 else y):
            return result.fail("Transpose does not swap the two matrix dimensions.")
        return result

    def rewrite(self, op, x, y, **_):
        return op.Gemm(x, y, **{f"trans{'A' if self.position == 0 else 'B'}": 1})


class TransposeAMatMulToGemm(_TransposeMatMulToGemm):
    """Replace ``MatMul(Transpose(A), B)`` with ``Gemm(A, B, transA=1)``."""


class TransposeBMatMulToGemm(_TransposeMatMulToGemm):
    """Replace ``MatMul(A, Transpose(B))`` with ``Gemm(A, B, transB=1)``."""

    position = 1


class _TransposeGemmInput(RewriteRuleClassBase):
    position: ClassVar[int] = 0

    def __init__(self) -> None:
        input_name = "A" if self.position == 0 else "B"
        super().__init__(f"FoldTransposeIntoGemmInput{input_name}", remove_nodes=False)

    def pattern(self, op, x, y):
        if self.position == 0:
            return op.Gemm(
                op.Transpose(x, _outputs=["transposed"]),
                y,
                _allow_other_inputs=True,
                _allow_other_attributes=True,
                _outputs=["gemm"],
            )
        return op.Gemm(
            x,
            op.Transpose(y, _outputs=["transposed"]),
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gemm"],
        )

    def check(self, context, x, y, transposed, **_) -> MatchResult:
        result = MatchResult()
        if not (_ir_utils.has_rank(x, 2) and _ir_utils.has_rank(y, 2)):
            return result.fail("Gemm inputs must both have rank two.")
        transpose = transposed.producer()
        assert transpose is not None
        gemm = context.root
        if _has_reference_attribute(gemm):
            return result.fail("Gemm attributes must be concrete to fold a Transpose.")
        if not _matrix_transpose(transpose, x if self.position == 0 else y):
            return result.fail("Transpose does not swap the two matrix dimensions.")
        return result

    def rewrite(self, op, x, y, gemm, **_):
        node = gemm.producer()
        assert node is not None
        inputs = list(node.inputs)
        inputs[self.position] = x if self.position == 0 else y
        attributes = dict(node.attributes)
        name = f"trans{'A' if self.position == 0 else 'B'}"
        attributes[name] = 1 - node.attributes.get_int(name, 0)
        return op.Gemm(*inputs, **attributes)


class TransposeAGemmInput(_TransposeGemmInput):
    """Fold a transpose on Gemm input A into ``transA``."""


class TransposeBGemmInput(_TransposeGemmInput):
    """Fold a transpose on Gemm input B into ``transB``."""

    position = 1


class GemmOutputTranspose(RewriteRuleClassBase):
    """Fold a matrix transpose after an unbiased Gemm."""

    def __init__(self) -> None:
        super().__init__("FoldGemmOutputTranspose", remove_nodes=True)

    def pattern(self, op, x, y):
        gemm = op.Gemm(
            x,
            y,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gemm"],
        )
        return op.Transpose(gemm, _outputs=["transposed"])

    def check(self, context, x, y, gemm, transposed, **_) -> MatchResult:
        result = MatchResult()
        gemm_node = gemm.producer()
        transpose = transposed.producer()
        assert gemm_node is not None and transpose is not None
        if not _is_private_intermediate(context, gemm):
            return result.fail("The Gemm output must only feed the Transpose.")
        if _has_reference_attribute(gemm_node):
            return result.fail(
                "Gemm attributes must be concrete to fold its output transpose."
            )
        if len(gemm_node.inputs) > 2 and gemm_node.inputs[2] is not None:
            return result.fail("A Gemm bias cannot be transposed without an extra operation.")
        if not (_ir_utils.has_rank(x, 2) and _ir_utils.has_rank(y, 2)):
            return result.fail("Gemm inputs must both have rank two.")
        if not _matrix_transpose(transpose, gemm):
            return result.fail("Transpose does not swap the two matrix dimensions.")
        return result

    def rewrite(self, op, x, y, gemm, **_):
        node = gemm.producer()
        assert node is not None
        attributes = dict(node.attributes)
        trans_a = node.attributes.get_int("transA", 0)
        trans_b = node.attributes.get_int("transB", 0)
        attributes["transA"] = 1 - trans_b
        attributes["transB"] = 1 - trans_a
        return op.Gemm(y, x, **attributes)


class _GemmSumBias(RewriteRuleClassBase):
    gemm_first: ClassVar[bool] = True

    def __init__(self) -> None:
        order = "GemmSum" if self.gemm_first else "SumGemm"
        super().__init__(f"Fuse{order}Bias", remove_nodes=True)

    def pattern(self, op, x, y, bias):
        gemm = op.Gemm(
            x,
            y,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gemm"],
        )
        if self.gemm_first:
            return op.Sum(gemm, bias)
        return op.Sum(bias, gemm)

    def check(self, context, x, y, bias, gemm, **_) -> MatchResult:
        result = MatchResult()
        node = gemm.producer()
        assert node is not None
        if not _is_private_intermediate(context, gemm):
            return result.fail("The Gemm output must only feed the Sum.")
        if _has_reference_attribute(node):
            return result.fail("Gemm attributes must be concrete to absorb the Sum bias.")
        if len(node.inputs) > 2 and node.inputs[2] is not None:
            return result.fail("Only an unbiased Gemm can absorb a Sum input.")
        if not (_ir_utils.has_rank(x, 2) and _ir_utils.has_rank(y, 2)):
            return result.fail("Gemm inputs must both have rank two.")
        trans_a = node.attributes.get_int("transA", 0)
        trans_b = node.attributes.get_int("transB", 0)
        m = _ir_utils.get_dim(x, 1 if trans_a else 0)
        n = _ir_utils.get_dim(y, 0 if trans_b else 1)
        if bias.shape is None or bias.shape.rank() > 2:
            return result.fail("The Sum bias must have known rank at most two.")
        for bias_dim, output_dim in zip(reversed(bias.shape.dims), (n, m)):
            if bias_dim != 1 and not _ir_utils.same_dim(bias_dim, output_dim):
                return result.fail("The Sum input is not a valid Gemm bias.")
        return result

    def rewrite(self, op, x, y, bias, gemm, **_):
        attributes = _node_attributes(gemm)
        attributes["beta"] = 1.0
        return op.Gemm(x, y, bias, **attributes)


class GemmSumBias(_GemmSumBias):
    """Replace ``Sum(Gemm(A, B), C)`` with ``Gemm(A, B, C)``."""


class SumBiasGemm(_GemmSumBias):
    """Replace ``Sum(C, Gemm(A, B))`` with ``Gemm(A, B, C)``."""

    gemm_first = False


class MatMulUnitReductionToMul(RewriteRuleClassBase):
    """Replace MatMul by broadcasting Mul when its reduction dimension is one."""

    def pattern(self, op, x, y):
        return op.MatMul(x, y)

    def check(self, context, x, y) -> MatchResult:
        del context
        result = MatchResult()
        if x.dtype not in _EXACT_UNIT_REDUCTION_TYPES or y.dtype != x.dtype:
            return result.fail(
                "Only integer MatMul is exactly equivalent to Mul for a unit reduction."
            )
        if x.shape is None or y.shape is None or x.shape.rank() < 2 or y.shape.rank() < 2:
            return result.fail("Both MatMul inputs must have known rank at least two.")
        if x.shape[-1] != 1 or y.shape[-2] != 1:
            return result.fail("Both MatMul reduction dimensions must equal one.")
        return result

    def rewrite(self, op, x, y):
        return op.Mul(x, y)


def _static_shape(value: ir.Value) -> list[int] | None:
    if value.shape is None or any(not isinstance(dim, int) for dim in value.shape):
        return None
    return list(value.shape)


class _ExpandBroadcastMatMul(RewriteRuleClassBase):
    expand_a: ClassVar[bool] = False
    expand_b: ClassVar[bool] = False

    def __init__(self) -> None:
        inputs = ("A" if self.expand_a else "") + ("B" if self.expand_b else "")
        super().__init__(f"RemoveRedundantExpandFromMatMulInput{inputs}", remove_nodes=False)

    def pattern(self, op, x, y, shape_a=None, shape_b=None):
        left = op.Expand(x, shape_a) if self.expand_a else x
        right = op.Expand(y, shape_b) if self.expand_b else y
        return op.MatMul(left, right)

    def check(self, context, x, y, **_) -> MatchResult:
        result = MatchResult()
        x_shape = _static_shape(x)
        y_shape = _static_shape(y)
        output_shape = _static_shape(context.root.outputs[0])
        if x_shape is None or y_shape is None or output_shape is None:
            return result.fail("MatMul input and output shapes must be fully static.")
        direct_shape = _broadcast_to_matmul._compute_matmul_output_shape(x_shape, y_shape)
        if direct_shape != output_shape:
            return result.fail(
                "The unexpanded inputs do not produce the original MatMul output shape."
            )
        return result

    def rewrite(self, op, x, y, **_):
        return op.MatMul(x, y)


class ExpandAMatMul(_ExpandBroadcastMatMul):
    """Remove an Expand on MatMul input A when MatMul broadcasting is equivalent."""

    expand_a = True


class ExpandBMatMul(_ExpandBroadcastMatMul):
    """Remove an Expand on MatMul input B when MatMul broadcasting is equivalent."""

    expand_b = True


class ExpandABMatMul(_ExpandBroadcastMatMul):
    """Remove Expands on both MatMul inputs when MatMul broadcasting is equivalent."""

    expand_a = True
    expand_b = True


class ReshapeABMatMul(RewriteRuleClassBase):
    """Remove two pre-MatMul reshapes when they preserve the direct MatMul result."""

    def __init__(self) -> None:
        super().__init__("RemoveRedundantReshapesFromMatMulInputs", remove_nodes=False)

    def pattern(self, op, x, y, shape_a, shape_b):
        return op.MatMul(op.Reshape(x, shape_a), op.Reshape(y, shape_b))

    def check(self, context, x, y, shape_a, shape_b) -> MatchResult:
        result = MatchResult()
        x_shape = _static_shape(x)
        y_shape = _static_shape(y)
        output_shape = _static_shape(context.root.outputs[0])
        shape_a_value = _ir_utils.get_numpy_value(shape_a)
        shape_b_value = _ir_utils.get_numpy_value(shape_b)
        if (
            x_shape is None
            or y_shape is None
            or output_shape is None
            or shape_a_value is None
            or shape_b_value is None
            or shape_a_value.ndim != 1
            or shape_b_value.ndim != 1
        ):
            return result.fail("Reshape targets and MatMul shapes must be fully static.")
        reshape_a = context.root.inputs[0].producer()
        reshape_b = context.root.inputs[1].producer()
        allowzero_a = _broadcast_to_matmul._get_allowzero(reshape_a)
        allowzero_b = _broadcast_to_matmul._get_allowzero(reshape_b)
        if not _broadcast_to_matmul._validate_reshape_layout(
            x_shape, shape_a_value.tolist(), allowzero=allowzero_a
        ):
            return result.fail("The first Reshape changes matrix or batch-axis layout.")
        if not _broadcast_to_matmul._validate_reshape_layout(
            y_shape, shape_b_value.tolist(), is_input_b=True, allowzero=allowzero_b
        ):
            return result.fail("The second Reshape changes matrix or batch-axis layout.")
        if _broadcast_to_matmul._compute_matmul_output_shape(x_shape, y_shape) != output_shape:
            return result.fail("Direct MatMul does not preserve the original output shape.")
        return result

    def rewrite(self, op, x, y, **_):
        return op.MatMul(x, y)


class MaxToRelu(RewriteRuleClassBase):
    """Replace ``Max(x, +0)`` with Relu while preserving NaN and signed-zero behavior."""

    def pattern(self, op, x, zero):
        return op.Max(x, zero)

    def check(self, context, x, zero) -> MatchResult:
        result = MatchResult()
        value = _ir_utils.get_numpy_value(zero)
        if value is None or value.ndim != 0 or value.item() != 0:
            return result.fail("The second Max input must be a scalar zero.")
        if zero.dtype != x.dtype:
            return result.fail("Max inputs must have the same type.")
        if np.signbit(value.item()):
            return result.fail("The scalar zero must not be negative zero.")
        supported_types = {
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
        }
        opset = _main_opset(context) or 0
        if opset >= 13:
            supported_types.add(ir.DataType.BFLOAT16)
        if opset >= 14:
            supported_types.update(
                {
                    ir.DataType.INT8,
                    ir.DataType.INT16,
                    ir.DataType.INT32,
                    ir.DataType.INT64,
                }
            )
        if x.dtype not in supported_types:
            return result.fail("The input type is not supported by Relu at this opset.")
        return result

    def rewrite(self, op, x, **_):
        return op.Relu(x)


class WhereToLeakyRelu(RewriteRuleClassBase):
    """Fuse the exact ``Where(x > 0, x, x * alpha)`` LeakyRelu decomposition."""

    def pattern(self, op, x, zero, slope):
        return op.Where(op.Greater(x, zero), x, op.Mul(x, slope))

    def check(self, context, x, zero, slope) -> MatchResult:
        result = MatchResult()
        opset = _main_opset(context) or 0
        if opset < 6:
            return result.fail("LeakyRelu requires ONNX opset 6 or newer.")
        if x.dtype not in {
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.BFLOAT16,
        }:
            return result.fail("LeakyRelu requires a floating-point input.")
        if x.dtype == ir.DataType.BFLOAT16 and opset < 16:
            return result.fail("LeakyRelu supports BFLOAT16 starting in ONNX opset 16.")
        if zero.dtype != x.dtype or slope.dtype != x.dtype:
            return result.fail("The decomposition inputs must have the same type.")
        zero_value = _ir_utils.get_numpy_value(zero)
        slope_value = _ir_utils.get_numpy_value(slope)
        if zero_value is None or zero_value.ndim != 0 or zero_value.item() != 0:
            return result.fail("The Greater threshold must be scalar zero.")
        if slope_value is None or slope_value.ndim != 0:
            return result.fail("The LeakyRelu slope must be scalar.")
        alpha = float(slope_value.item())
        if not np.isfinite(alpha) or alpha < 0 or np.signbit(alpha):
            return result.fail("The slope must be finite and nonnegative.")
        if float(np.float32(alpha)) != alpha:
            return result.fail("The slope is not exactly representable by a FLOAT attribute.")
        self._alpha = alpha
        return result

    def rewrite(self, op, x, **_):
        return op.LeakyRelu(x, alpha=self._alpha)


def _constant_axes(value: ir.Value, expected: int) -> bool:
    array = _ir_utils.get_numpy_value(value)
    if array is None or array.ndim != 1 or array.size != 1:
        return False
    axis = int(array.item())
    return axis == expected or axis == expected - 4


def _concrete_perm(value: ir.Value, expected: tuple[int, ...]) -> bool:
    node = value.producer()
    if node is None or _has_reference_attribute(node, {"perm"}):
        return False
    return tuple(node.attributes.get_ints("perm")) == expected


class LinearAttentionT1(RewriteRuleClassBase):
    """Fuse the exact float32, equal-head, single-token linear recurrence."""

    def __init__(self) -> None:
        super().__init__("FuseSingleTokenLinearAttention")

    def pattern(
        self,
        op,
        query,
        key,
        value,
        past_state,
        query_shape,
        key_shape,
        value_shape,
        output_shape,
        unpack_axes,
        key_axes,
        value_axes,
        query_axes,
        read_axes,
        pack_axes,
        scale,
    ):
        query_4d = op.Reshape(query, query_shape, _outputs=["query_4d"])
        query_transposed = op.Transpose(
            query_4d,
            _allow_other_attributes=True,
            _outputs=["query_transposed"],
        )
        query_unpacked = op.Squeeze(query_transposed, unpack_axes, _outputs=["query_unpacked"])
        key_4d = op.Reshape(key, key_shape, _outputs=["key_4d"])
        key_transposed = op.Transpose(
            key_4d,
            _allow_other_attributes=True,
            _outputs=["key_transposed"],
        )
        key_unpacked = op.Squeeze(key_transposed, unpack_axes, _outputs=["key_unpacked"])
        value_4d = op.Reshape(value, value_shape, _outputs=["value_4d"])
        value_transposed = op.Transpose(
            value_4d,
            _allow_other_attributes=True,
            _outputs=["value_transposed"],
        )
        value_unpacked = op.Squeeze(value_transposed, unpack_axes, _outputs=["value_unpacked"])
        outer = op.Mul(
            op.Unsqueeze(key_unpacked, key_axes),
            op.Unsqueeze(value_unpacked, value_axes),
        )
        present_state = op.Add(past_state, outer, _outputs=["present_state"])
        read = op.MatMul(
            op.Unsqueeze(query_unpacked, query_axes),
            present_state,
        )
        read = op.Squeeze(read, read_axes)
        scaled = op.Mul(read, scale)
        packed = op.Unsqueeze(scaled, pack_axes)
        packed = op.Transpose(
            packed,
            _allow_other_attributes=True,
            _outputs=["output_transposed"],
        )
        output = op.Reshape(packed, output_shape)
        return output, present_state

    def check(
        self,
        context,
        query,
        key,
        value,
        past_state,
        query_shape,
        key_shape,
        value_shape,
        output_shape,
        unpack_axes,
        key_axes,
        value_axes,
        query_axes,
        read_axes,
        pack_axes,
        scale,
        query_transposed,
        key_transposed,
        value_transposed,
        output_transposed,
        **_,
    ) -> MatchResult:
        result = MatchResult()
        if (_main_opset(context) or 0) < 27:
            return result.fail("LinearAttention requires ONNX opset 27 or newer.")
        if any(
            input_value.dtype != ir.DataType.FLOAT
            for input_value in (query, key, value, past_state)
        ):
            return result.fail(
                "Only FLOAT preserves the decomposition's accumulation precision."
            )
        query_dims = _static_shape(query)
        key_dims = _static_shape(key)
        value_dims = _static_shape(value)
        state_dims = _static_shape(past_state)
        if (
            query_dims is None
            or key_dims is None
            or value_dims is None
            or state_dims is None
            or len(query_dims) != 3
            or len(key_dims) != 3
            or len(value_dims) != 3
            or len(state_dims) != 4
            or query_dims[1] != 1
            or key_dims[:2] != query_dims[:2]
            or value_dims[:2] != query_dims[:2]
        ):
            return result.fail("Packed inputs must have static [B, 1, H*D] shapes.")
        reshape_shapes = []
        for shape_value in (query_shape, key_shape, value_shape):
            array = _ir_utils.get_numpy_value(shape_value)
            if array is None or array.ndim != 1 or array.size != 4:
                return result.fail("Unpack Reshape targets must be constant rank-four shapes.")
            reshape_shapes.append([int(dim) for dim in array])
        query_reshape, key_reshape, value_reshape = reshape_shapes
        heads = query_reshape[2]
        if (
            heads <= 0
            or key_reshape[2] != heads
            or value_reshape[2] != heads
            or query_reshape != [query_dims[0], 1, heads, state_dims[2]]
            or key_reshape != [key_dims[0], 1, heads, state_dims[2]]
            or value_reshape != [value_dims[0], 1, heads, state_dims[3]]
            or state_dims != [query_dims[0], heads, state_dims[2], state_dims[3]]
            or query_dims[2] != heads * state_dims[2]
            or key_dims[2] != heads * state_dims[2]
            or value_dims[2] != heads * state_dims[3]
        ):
            return result.fail("Query, key, value, and state must use the same head count.")
        output_shape_array = _ir_utils.get_numpy_value(output_shape)
        if (
            output_shape_array is None
            or output_shape_array.ndim != 1
            or output_shape_array.tolist() != [query_dims[0], 1, heads * state_dims[3]]
        ):
            return result.fail("The output Reshape target is not canonical packed form.")
        if not (
            _constant_axes(unpack_axes, 2)
            and _constant_axes(key_axes, 3)
            and _constant_axes(value_axes, 2)
            and _constant_axes(query_axes, 2)
            and _constant_axes(read_axes, 2)
            and _constant_axes(pack_axes, 2)
        ):
            return result.fail("LinearAttention layout axes are not canonical.")
        if not all(
            _concrete_perm(transposed, (0, 2, 1, 3))
            for transposed in (
                query_transposed,
                key_transposed,
                value_transposed,
                output_transposed,
            )
        ):
            return result.fail("LinearAttention layout transposes are not canonical.")
        scale_array = _ir_utils.get_numpy_value(scale)
        if scale.dtype != ir.DataType.FLOAT or scale_array is None or scale_array.ndim != 0:
            return result.fail("The output scale must be a scalar FLOAT constant.")
        scale_value = float(scale_array.item())
        if not np.isfinite(scale_value) or scale_value == 0:
            return result.fail("The output scale must be finite and nonzero.")
        self._heads = heads
        self._scale = scale_value
        return result

    def rewrite(self, op, query, key, value, past_state, **_):
        return op.LinearAttention(
            query,
            key,
            value,
            past_state,
            kv_num_heads=self._heads,
            q_num_heads=self._heads,
            scale=self._scale,
            update_rule="linear",
            _outputs=2,
        )


_LAYOUT_MOVABLE_ACTIVATIONS = (
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Gelu",
    "LeakyRelu",
    "Relu",
    "Selu",
    "Sin",
    "Sinh",
    "Softplus",
    "Tan",
    "Tanh",
)


class SwitchLayoutActivation(RewriteRuleClassBase):
    """Move an elementwise activation before a private Reshape or Transpose."""

    def __init__(self, activation: str, layout: str):
        self.activation = activation
        self.layout = layout
        super().__init__(f"Move{activation}Before{layout}", remove_nodes=False)

    def pattern(self, op, product, shape=None):
        if self.layout == "Reshape":
            layout = op.Reshape(
                product,
                shape,
                _allow_other_attributes=True,
                _outputs=["layout_output"],
            )
        else:
            layout = op.Transpose(
                product,
                _allow_other_attributes=True,
                _outputs=["layout_output"],
            )
        return getattr(op, self.activation)(
            layout,
            _allow_other_attributes=True,
            _outputs=["activation_output"],
        )

    def check(self, context, product, layout_output, activation_output, **_) -> MatchResult:
        result = MatchResult()
        producer = product.producer()
        layout = layout_output.producer()
        activation = activation_output.producer()
        assert layout is not None and activation is not None
        if (
            producer is None
            or producer.domain not in {"", "ai.onnx"}
            or producer.op_type not in {"MatMul", "Gemm"}
        ):
            return result.fail("The layout input must be produced by MatMul or Gemm.")
        if not _is_private_intermediate(context, layout_output):
            return result.fail("The layout output must only feed the activation.")
        if _has_reference_attribute(layout) or _has_reference_attribute(activation):
            return result.fail("Layout and activation attributes must be concrete.")
        return result

    def rewrite(self, op, product, layout_output, activation_output, shape=None, **_):
        activated = getattr(op, self.activation)(
            product, **_node_attributes(activation_output)
        )
        if self.layout == "Reshape":
            return op.Reshape(activated, shape, **_node_attributes(layout_output))
        return op.Transpose(activated, **_node_attributes(layout_output))


class _GroupedMatMulBase(RewriteRuleClassBase):
    with_bias: ClassVar[bool] = False
    add_reversed: ClassVar[bool] = False

    def __init__(self) -> None:
        if not self.with_bias:
            name = "FuseGroupedMatMulDecomposition"
        elif self.add_reversed:
            name = "FuseReversedBiasGroupedMatMulDecomposition"
        else:
            name = "FuseGroupedMatMulBiasDecomposition"
        super().__init__(name, remove_nodes=False)

    def _product(self, op, x, weights, indices, unsqueeze_axes):
        selected = op.Gather(
            weights,
            indices,
            _allow_other_attributes=True,
            _outputs=["selected"],
        )
        rows = op.Unsqueeze(x, unsqueeze_axes, _outputs=["rows"])
        return op.MatMul(rows, selected, _outputs=["product"])

    def check(
        self,
        context,
        x,
        weights,
        indices,
        unsqueeze_axes,
        squeeze_axes,
        bias=None,
        selected=None,
        selected_bias=None,
        **_,
    ) -> MatchResult:
        result = MatchResult()
        if (_main_opset(context) or 0) < 29:
            return result.fail("GroupedMatMul requires ONNX opset 29 or newer.")
        if any(
            node is not context.root
            and node.op_type in {"MatMul", "Squeeze"}
            and not _is_private_intermediate(context, node.outputs[0])
            for node in context.nodes
        ):
            return result.fail("Grouped-matrix fusion must not duplicate shared matrix work.")
        if not _ir_utils.is_1d_value(unsqueeze_axes, [1, 2]):
            return result.fail("The row input must be unsqueezed on axes [1, 2].")
        if not _ir_utils.is_1d_value(squeeze_axes, [2]):
            return result.fail("The MatMul result must be squeezed on axis [2].")
        for gathered in (selected, selected_bias):
            if gathered is None:
                continue
            gather = gathered.producer()
            assert gather is not None
            if _has_reference_attribute(gather, {"axis"}):
                return result.fail("Gather axis attributes must be concrete.")
            if gather.attributes.get_int("axis", 0) != 0:
                return result.fail("GroupedMatMul gathers must select on axis zero.")
        if x.shape is None or weights.shape is None or indices.shape is None:
            return result.fail("GroupedMatMul input shapes must be known.")
        if x.shape.rank() != 2 or weights.shape.rank() != 3 or indices.shape.rank() != 2:
            return result.fail("Expected input, weights, indices ranks [2, 3, 2].")
        if not (
            _ir_utils.same_dim(x.shape[0], indices.shape[0])
            and _ir_utils.same_dim(x.shape[1], weights.shape[1])
        ):
            return result.fail("GroupedMatMul dimensions are incompatible.")
        if not isinstance(weights.shape[0], int) or weights.shape[0] <= 0:
            return result.fail("The number of groups must be a positive static dimension.")
        if indices.dtype not in (ir.DataType.INT32, ir.DataType.INT64):
            return result.fail("Group indices must be INT32 or INT64.")
        if indices.dtype == ir.DataType.INT32 and weights.shape[0] > np.iinfo(np.int32).max:
            return result.fail("The number of groups is not representable by INT32 indices.")
        if x.dtype != weights.dtype or x.dtype not in _GEMM_TYPES:
            return result.fail(
                "GroupedMatMul data and weight types must match and be floating."
            )
        if bias is not None:
            if (
                bias.shape is None
                or bias.shape.rank() != 2
                or bias.dtype != x.dtype
                or not _ir_utils.same_dim(bias.shape[0], weights.shape[0])
                or not _ir_utils.same_dim(bias.shape[1], weights.shape[2])
            ):
                return result.fail("GroupedMatMul bias must have shape [groups, output_size].")
        self._groups = weights.shape[0]
        self._index_dtype = indices.dtype
        return result

    def rewrite(self, op, x, weights, indices, bias=None, **_):
        np_dtype = np.int32 if self._index_dtype == ir.DataType.INT32 else np.int64
        zero = op.Constant(value=ir.tensor(np.array(0, dtype=np_dtype)))
        groups = op.Constant(value=ir.tensor(np.array(self._groups, dtype=np_dtype)))
        normalized = op.Where(op.Less(indices, zero), op.Add(indices, groups), indices)
        inputs = [x, weights, normalized]
        if bias is not None:
            inputs.append(bias)
        return op.GroupedMatMul(*inputs)


class GroupedMatMul(_GroupedMatMulBase):
    """Fuse the canonical Gather/MatMul grouped-expert decomposition."""

    def pattern(self, op, x, weights, indices, unsqueeze_axes, squeeze_axes):
        return op.Squeeze(
            self._product(op, x, weights, indices, unsqueeze_axes),
            squeeze_axes,
        )


class GroupedMatMulBias(_GroupedMatMulBase):
    """Fuse the grouped-expert decomposition with a gathered bias."""

    with_bias = True

    def pattern(self, op, x, weights, indices, unsqueeze_axes, squeeze_axes, bias):
        output = op.Squeeze(
            self._product(op, x, weights, indices, unsqueeze_axes),
            squeeze_axes,
        )
        selected_bias = op.Gather(
            bias,
            indices,
            _allow_other_attributes=True,
            _outputs=["selected_bias"],
        )
        return op.Add(output, selected_bias)


class GroupedMatMulBiasReversed(GroupedMatMulBias):
    """Fuse the grouped-expert decomposition with reversed Add operands."""

    add_reversed = True

    def pattern(self, op, x, weights, indices, unsqueeze_axes, squeeze_axes, bias):
        output = op.Squeeze(
            self._product(op, x, weights, indices, unsqueeze_axes),
            squeeze_axes,
        )
        selected_bias = op.Gather(
            bias,
            indices,
            _allow_other_attributes=True,
            _outputs=["selected_bias"],
        )
        return op.Add(selected_bias, output)


class _GroupedMatMulGatherBias(RewriteRuleClassBase):
    add_reversed: ClassVar[bool] = False

    def __init__(self) -> None:
        order = "ReversedGatheredBias" if self.add_reversed else "GatheredBias"
        super().__init__(f"Fuse{order}IntoGroupedMatMul", remove_nodes=False)

    def pattern(self, op, x, weights, indices, zero, groups, bias):
        normalized = op.Where(
            op.Less(indices, zero),
            op.Add(indices, groups),
            indices,
            _outputs=["normalized"],
        )
        grouped = op.GroupedMatMul(x, weights, normalized, _outputs=["grouped"])
        selected_bias = op.Gather(
            bias,
            indices,
            _allow_other_attributes=True,
            _outputs=["selected_bias"],
        )
        if self.add_reversed:
            return op.Add(selected_bias, grouped)
        return op.Add(grouped, selected_bias)

    def check(
        self,
        context,
        x,
        weights,
        indices,
        bias,
        zero,
        groups,
        grouped,
        selected_bias,
        **_,
    ) -> MatchResult:
        result = MatchResult()
        if (_main_opset(context) or 0) < 29:
            return result.fail("GroupedMatMul requires ONNX opset 29 or newer.")
        if weights.shape is None or bias.shape is None:
            return result.fail("GroupedMatMul weight and bias shapes must be known.")
        if not _is_private_intermediate(context, grouped):
            return result.fail("The unbiased GroupedMatMul result must not be shared.")
        gather = selected_bias.producer()
        assert gather is not None
        if _has_reference_attribute(gather, {"axis"}):
            return result.fail("Gather axis attributes must be concrete.")
        if gather.attributes.get_int("axis", 0) != 0:
            return result.fail("The bias Gather must select on axis zero.")
        if (
            weights.shape.rank() != 3
            or bias.shape.rank() != 2
            or bias.dtype != x.dtype
            or not _ir_utils.same_dim(bias.shape[0], weights.shape[0])
            or not _ir_utils.same_dim(bias.shape[1], weights.shape[2])
        ):
            return result.fail("GroupedMatMul bias must have shape [groups, output_size].")
        if not (
            isinstance(weights.shape[0], int)
            and weights.shape[0] > 0
            and indices.dtype in (ir.DataType.INT32, ir.DataType.INT64)
            and zero.dtype == indices.dtype
            and groups.dtype == indices.dtype
            and _ir_utils.is_singleton_value(zero, 0)
            and _ir_utils.is_singleton_value(groups, weights.shape[0])
        ):
            return result.fail(
                "Gather and GroupedMatMul must use identical index normalization."
            )
        return result

    def rewrite(self, op, x, weights, bias, normalized, **_):
        return op.GroupedMatMul(x, weights, normalized, bias)


class GroupedMatMulGatherBias(_GroupedMatMulGatherBias):
    """Absorb a gathered bias after grouped-matrix fusion."""


class GroupedMatMulGatherBiasReversed(_GroupedMatMulGatherBias):
    """Absorb a gathered bias with reversed Add operands."""

    add_reversed = True


rules = RewriteRuleSet(
    [
        ExpandABMatMul.rule(),
        ExpandAMatMul.rule(),
        ExpandBMatMul.rule(),
        ReshapeABMatMul.rule(),
        GroupedMatMulBias.rule(),
        GroupedMatMulBiasReversed.rule(),
        GroupedMatMul.rule(),
        GroupedMatMulGatherBias.rule(),
        GroupedMatMulGatherBiasReversed.rule(),
        GemmOutputTranspose.rule(),
        TransposeAGemmInput.rule(),
        TransposeBGemmInput.rule(),
        TransposeAMatMulToGemm.rule(),
        TransposeBMatMulToGemm.rule(),
        GemmSumBias.rule(),
        SumBiasGemm.rule(),
        MatMulUnitReductionToMul.rule(name="ReplaceUnitReductionMatMulWithMul"),
        MaxToRelu.rule(name="ReplaceMaxWithRelu"),
        WhereToLeakyRelu.rule(name="FuseWhereMulIntoLeakyRelu"),
        LinearAttentionT1.rule(),
        *(
            SwitchLayoutActivation.rule(activation, layout)
            for activation in _LAYOUT_MOVABLE_ACTIVATIONS
            for layout in ("Reshape", "Transpose")
        ),
    ]
)
