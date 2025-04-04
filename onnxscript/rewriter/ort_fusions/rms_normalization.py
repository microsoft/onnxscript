# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

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


class RmsNormFusion(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, *, cast_input: bool, cast_normalized: bool):
        """
        Args:
            name: Name of the rule.
            cast_input: Whether to cast input to do the normalization in a different precision.
            cast_normalized: Whether to cast the normalized output to the target dtype (same as scale).
        """
        super().__init__(name=name)
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

    def check(self, op, x, scale, epsilon, compute_dtype, target_dtype) -> pattern.MatchResult:  # type: ignore[name-defined]
        """Check if the pattern matches conditions for use of SimplifiedLayerNormalization op."""
        check_result = pattern.MatchResult()
        # epsilon must be a scalar
        epsilon_value = _ir_utils.get_singleton_value(epsilon)
        if not isinstance(epsilon_value, float):  # TODO: support other types
            return check_result.fail("Epsilon is not a float value.", epsilon)
        # input and output must be same dtype
        if x.dtype not in float_types:
            return check_result.fail("Input is not a float type.", x)
        if scale.dtype not in float_types:
            return check_result.fail("Scale is not a float type.", scale)
        stash_dtype = compute_dtype.value if self._cast_input else x.dtype
        if stash_dtype not in fp_float_types:
            return check_result.fail("Normalization precision is not a float or double type.")
        return check_result

    def rewrite(self, op, x, scale, epsilon, compute_dtype, target_dtype):
        stash_dtype = compute_dtype.value if self._cast_input else x.dtype
        # Note: ORT's SimplifiedLayerNormalization was placed in onnx domain by mistake.
        # No need to use com.microsoft domain here; but this is a custom op in ORT.
        return op.SimplifiedLayerNormalization(
            x,
            scale,
            axis=-1,
            epsilon=_ir_utils.get_singleton_value(epsilon),
            stash_type=stash_dtype,
        )


_rule_0 = RmsNormFusion.rule("RmsNorm-0", cast_input=True, cast_normalized=True)
_rule_1 = RmsNormFusion.rule("RmsNorm-1", cast_input=False, cast_normalized=True)
_rule_2 = RmsNormFusion.rule("RmsNorm-2", cast_input=True, cast_normalized=False)
_rule_3 = RmsNormFusion.rule("RmsNorm-3", cast_input=False, cast_normalized=False)

rms_normalization_rules = [_rule_0, _rule_1, _rule_2, _rule_3]
rms_normalization_ruleset = pattern.RewriteRuleSet(rms_normalization_rules)


fuse_rms_normalization = _fusion_utils.apply_fusion_rules(rms_normalization_ruleset)
