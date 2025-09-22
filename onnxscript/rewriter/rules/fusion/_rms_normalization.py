# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

"""
RMS Normalization: ONNX Opset 23 op
See: https://onnx.ai/onnx/operators/onnx__RMSNormalization.html#l-onnx-doc-rmsnormalization


Key points for the fusion optimization:
* Input and scale are allowed to be of different types.
* The normalization of the input can be done in a different precision than the input type,
indicated by stash_type.
* Input (x) must be: float or double or float16 or bfloat16
* Scale must be: float or double or float16 or bfloat16
"""

float_types = frozenset(
    [
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.DOUBLE,
    ]
)
fp_float_types = frozenset([ir.DataType.FLOAT, ir.DataType.DOUBLE])


class RmsNormFusion(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, mul_order: bool):
        super().__init__(name)
        self._mul_order = mul_order

    def pattern(self, op, x, scale, epsilon, compute_dtype, target_dtype):
        x = pattern.OrValue([op.Cast(x, to=compute_dtype), x])
        x_square = op.Pow(x, 2.0)
        mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
        mean_square_plus_epsilon = op.Add(mean_square, epsilon)
        rms = op.Sqrt(mean_square_plus_epsilon)
        reciprocal_rms = op.Reciprocal(rms)
        normalized = op.Mul(x, reciprocal_rms)
        normalized = pattern.OrValue([op.Cast(normalized, to=target_dtype), normalized])
        # Workaround: limitation in pattern matcher doesn't support OrValue for return value (last node in pattern)
        if self._mul_order:
            return op.Mul(normalized, scale)
        else:
            return op.Mul(scale, normalized)

    def check(
        self, op, x, scale, epsilon, compute_dtype, target_dtype, **_
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        """Check if the pattern matches conditions for use of SimplifiedLayerNormalization op."""
        check_result = pattern.MatchResult()
        # epsilon must be a scalar
        epsilon_value = _ir_utils.get_singleton_value(epsilon)
        if not isinstance(epsilon_value, float):  # TODO: support other types
            return check_result.fail("Epsilon is not a float value.", epsilon)
        if x.dtype not in float_types:
            return check_result.fail("Input is not a supported float type.", x)
        if scale.dtype not in float_types:
            return check_result.fail("Scale is not a supported float type.", scale)
        self._stash_dtype = compute_dtype.as_int() if compute_dtype is not None else x.dtype
        if self._stash_dtype not in fp_float_types:
            # TODO: ONNX documentation does not specify restrictions on stash_type, though
            # ORT's SimplifiedLayerNormalization requires it to be float or double.
            return check_result.fail("Normalization precision is not a float or double type.")
        # target_dtype is guaranteed to be the same as scale type in a well-typed input
        # for Mul(scale, normalized) to work. There is no need to check it here for a well-typed input.
        # TODO (rama): Consider adding checks to protect against incorrectly typed models:
        return check_result

    def rewrite(self, op, x, scale, epsilon, **_):
        # Note: ORT's SimplifiedLayerNormalization was placed in onnx domain by mistake.
        # No need to use com.microsoft domain here; but this is a custom op in ORT.
        return op.RMSNormalization(
            x,
            scale,
            axis=-1,
            epsilon=_ir_utils.get_singleton_value(epsilon),
            stash_type=self._stash_dtype,
        )


_rule1 = RmsNormFusion.rule("RmsNormFusion1", mul_order=True)
_rule2 = RmsNormFusion.rule("RmsNormFusion2", mul_order=False)

rms_normalization_rules = [_rule1, _rule2]

rms_normalization_ruleset = pattern.RewriteRuleSet(rms_normalization_rules)

fuse_rms_normalization = _fusion_utils.apply_fusion_rules(rms_normalization_ruleset)
