# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

"""
RMS Normalization: This is referred to as SimplifiedLayerNormalization in the ORT codebase.
See https://github.com/microsoft/onnxruntime/blob/6d9636f07cccdb6e4ac453087ad54c3bc9854d50/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L2981

Key points for the fusion optimization:
* Input and scale are allowed to be of different types.
* The normalization of the input can be done in a different precision than the input type,
which is also the precision of reciprocal_rms returned by operation.
* Input (x) must be: float or double or float16 or bfloat16
* Scale must be: float or double or float16 or bfloat16
* Normalization precision must be float or double
"""

float_types = [
    ir.DataType.FLOAT,
    ir.DataType.FLOAT16,
    ir.DataType.BFLOAT16,
    ir.DataType.DOUBLE,
]
fp_float_types = [ir.DataType.FLOAT, ir.DataType.DOUBLE]


class RmsNormFusion:
    @classmethod
    def rule(cls, *, cast_input: bool, cast_normalized: bool):
        instance = cls(cast_input=cast_input, cast_normalized=cast_normalized)
        return pattern.RewriteRule(instance.pattern, instance.rewrite, instance.check)

    def __init__(self, *, cast_input: bool, cast_normalized: bool):
        """
        Args:
            cast_input: Whether to cast input to do the normalization in a different precision.
            cast_normalized: Whether to cast the normalized output to the target dtype (same as scale).
        """
        self._cast_input = cast_input
        self._cast_normalized = cast_normalized

    def pattern(self, op, x, scale, epsilon, compute_dtype, target_dtype):
        if self._cast_input:
            x = op.Cast(x, to=compute_dtype)
        x_square = op.Pow(x, 2.0)
        mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
        mean_square_plus_epsilon = op.Add(mean_square, epsilon)
        rms = op.Sqrt(mean_square_plus_epsilon)
        reciprocal_rms = op.Reciprocal(rms)
        normalized = op.Mul(x, reciprocal_rms)
        if self._cast_normalized:
            normalized = op.Cast(normalized, to=target_dtype)
        return op.Mul(scale, normalized)

    def check(self, op, x, scale, epsilon, compute_dtype, target_dtype):
        """Check if the pattern matches conditions for use of SimplifiedLayerNormalization op."""
        # epsilon must be a scalar
        epsilon_value = _ir_utils.get_singleton_value(epsilon)
        if not isinstance(epsilon_value, float):  # TODO: support other types
            return False
        # input and output must be same dtype
        if x.dtype not in float_types:
            return False
        if scale.dtype not in float_types:
            return False
        stash_dtype = compute_dtype.value if self._cast_input else x.dtype
        if stash_dtype not in fp_float_types:
            return False
        return True

    def rewrite(self, op, x, scale, epsilon, compute_dtype, target_dtype):
        stash_dtype = compute_dtype.value if self._cast_input else x.dtype
        # Note: ORT's SimplifiedLayerNormalization was placed in onnx domain by mistake.
        # No need to use com.microsoft domain here.
        return op.SimplifiedLayerNormalization(
            x,
            scale,
            axis=-1,
            epsilon=_ir_utils.get_singleton_value(epsilon),
            stash_type=stash_dtype,
        )


# # Pattern to match against
# def _rms_norm_pattern(op, x, scale, epsilon, compute_dtype, target_dtype):
#     x_cast = op.Cast(x, to=compute_dtype)
#     x_square = op.Pow(x_cast, 2.0)
#     mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
#     mean_square_plus_epsilon = op.Add(mean_square, epsilon)
#     rms = op.Sqrt(mean_square_plus_epsilon)
#     reciprocal_rms = op.Reciprocal(rms)
#     normalized = op.Mul(x_cast, reciprocal_rms)
#     normalized_cast = op.Cast(normalized, to=target_dtype)
#     return op.Mul(scale, normalized_cast)


# # Replacement
# def _simplified_layer_norm(op, x, scale, epsilon, compute_dtype, target_dtype):
#     epsilon_value = _ir_utils.get_singleton_value(epsilon)
#     if not isinstance(epsilon_value, float):
#         return None
#     source_dtype = x.dtype
#     if source_dtype is None or source_dtype != target_dtype.value:
#         return None
#     return op.SimplifiedLayerNormalization(
#         x,
#         scale,
#         axis=-1,
#         epsilon=epsilon_value,
#         stash_type=compute_dtype.value,
#     )

# # Pattern to match against
# def _rms_norm_pattern_no_cast(op, x, scale, epsilon):
#     # x_cast = op.Cast(x, to=compute_dtype)
#     x_cast = x
#     x_square = op.Pow(x_cast, 2.0)
#     mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
#     mean_square_plus_epsilon = op.Add(mean_square, epsilon)
#     rms = op.Sqrt(mean_square_plus_epsilon)
#     reciprocal_rms = op.Reciprocal(rms)
#     normalized = op.Mul(x_cast, reciprocal_rms)
#     # normalized_cast = op.Cast(normalized, to=target_dtype)
#     normalized_cast = normalized
#     return op.Mul(scale, normalized_cast)


# Replacement
# def _simplified_layer_norm_no_cast(op, x, scale, epsilon):
#     epsilon_value = _ir_utils.get_singleton_value(epsilon)
#     if not isinstance(epsilon_value, float):
#         return None
#     source_dtype = x.dtype
#     if source_dtype is None:
#         return None
#     return op.SimplifiedLayerNormalization(
#         x,
#         scale,
#         axis=-1,
#         epsilon=epsilon_value,
#         stash_type=source_dtype.value,
#     )

_rule_0 = RmsNormFusion.rule(cast_input=True, cast_normalized=True)
_rule_1 = RmsNormFusion.rule(cast_input=False, cast_normalized=True)
_rule_2 = RmsNormFusion.rule(cast_input=True, cast_normalized=False)
_rule_3 = RmsNormFusion.rule(cast_input=False, cast_normalized=False)

rms_normalization_rules = [_rule_0, _rule_1, _rule_2, _rule_3]
rms_normalization_ruleset = pattern.RewriteRuleSet(rms_normalization_rules)


def fuse_rms_normalization(model: ir.Model) -> None:
    count = rms_normalization_ruleset.apply_to_model(model)
    print(f"RMS Normalization count: {count}")
