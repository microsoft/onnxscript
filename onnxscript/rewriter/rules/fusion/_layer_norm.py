# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

"""
Layer Normalization fusion optimization.

This module contains rewrite rules for fusing Layer Normalization patterns into the
ONNX LayerNormalization operator.

Layer Normalization performs normalization over the last D dimensions as specified by the axis.
The computation follows: Y = scale * (X - mean) / sqrt(variance + epsilon) + bias

Key points for the fusion optimization:
* Following restrictions from opset 17 LayerNormalization:
* Input, scale, and bias must be of same type T in {float16, bfloat16, float, double}
* The normalization can be done in a different precision than the input type (bfloat16 or float),
which is also the precision of the output mean/invstddev
"""

# input types permitted by LayerNormalization op (ONNX Opset 17)
LAYER_NORM_INPUT_TYPES = frozenset(
    [
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.DOUBLE,
    ]
)

# Compute types permitted by LayerNormalization op (ONNX Opset 17), aka stash_type.
LAYER_NORM_COMPUTE_TYPES = frozenset([ir.DataType.FLOAT, ir.DataType.DOUBLE])


class LayerNormFusion(pattern.RewriteRuleClassBase):
    """Fuse LayerNorm pattern into LayerNormalization op."""

    def pattern(self, op, x, scale, epsilon):
        # Compute mean: Mean = ReduceMean(X, axes=normalized_axes)
        # TODO: support axes attribute too
        mean = op.ReduceMean(x, [-1], keepdims=1)

        # Compute deviation: D = Sub(X, Mean)
        deviation = op.Sub(x, mean)

        # Compute squared deviation: DD = Mul(D, D)
        deviation_squared = pattern.OrValue(
            [
                op.Mul(deviation, deviation),
                op.Pow(deviation, 2),
            ]
        )

        # Compute variance: Var = ReduceMean(DD, axes=normalized_axes)
        variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)

        # Add epsilon: VarEps = Add(Var, epsilon)
        variance_plus_epsilon = op.Add(variance, epsilon)

        # Compute standard deviation: StdDev = Sqrt(VarEps)
        std_dev = op.Sqrt(variance_plus_epsilon)

        # Compute reciprocal: InvStdDev = Reciprocal(StdDev)
        # Normalize: Normalized = Mul(D, InvStdDev)

        inv_std_dev = op.Reciprocal(std_dev)
        normalized = pattern.OrValue(
            [op.Mul(deviation, inv_std_dev), op.Div(deviation, std_dev)]
        )

        # Scale: NormalizedScaled = Mul(Normalized, Scale)
        normalized_scaled = op.Mul(normalized, scale)

        return normalized_scaled

    def check(self, context, x, epsilon, **_) -> pattern.MatchResult:  # type: ignore[name-defined]
        """Check if the pattern matches conditions for use of LayerNormalization op."""
        check_result = pattern.MatchResult()

        # Type validation:
        if x.dtype not in LAYER_NORM_COMPUTE_TYPES:
            return check_result.fail("Input is not a float type.", x)
        self._stash_type = x.dtype

        # Check that epsilon is a scalar constant
        epsilon_value = _ir_utils.get_singleton_value(epsilon)
        if epsilon_value is None:
            return check_result.fail("Epsilon is not a constant scalar.", epsilon)
        # Epsilon is guaranteed to be same type as x (float or double, in this pattern)
        self._epsilon = float(epsilon_value)

        return check_result

    def rewrite(self, op, x, scale, epsilon, **_):
        return op.LayerNormalization(
            x,
            scale,
            axis=-1,
            epsilon=self._epsilon,
            stash_type=self._stash_type,
        )


class LayerNormBiasFusion(pattern.RewriteRuleClassBase):
    """Fuse LayerNorm => Add into LayerNorm with bias."""

    def pattern(self, op, x, scale, bias):
        return op.LayerNormalization(x, scale, _outputs=["normalized"]) + bias

    def rewrite(self, op, x, scale, bias, normalized):
        layernorm_node = normalized.producer()
        attributes = layernorm_node.attributes
        num_outputs = len(layernorm_node.outputs)
        return op.LayerNormalization(x, scale, bias, _outputs=num_outputs, **attributes)


# Create rules for both with and without bias
_layer_norm_rule = LayerNormFusion.rule()
_layer_norm_with_bias_rule = LayerNormBiasFusion.rule()

layer_normalization_rules = [_layer_norm_rule, _layer_norm_with_bias_rule]
layer_normalization_ruleset = pattern.RewriteRuleSet(layer_normalization_rules)

fuse_layer_normalization = _fusion_utils.apply_fusion_rules(layer_normalization_ruleset)
