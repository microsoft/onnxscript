# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, pattern


def _skip_rms_norm_pattern(op, input, skip, gamma, epsilon, stash_type):
    skip_sum = op.Add(input, skip)
    normalized = op.SimplifiedLayerNormalization(
        skip_sum,
        gamma,
        axis=-1,
        epsilon=epsilon,
        stash_type=stash_type,
    )
    return normalized, skip_sum


def _skip_rms_normalization(op, input, skip, gamma, epsilon, stash_type):
    if stash_type.value != 1:  # FLOAT type
        return None
    normalized, _mean, _inv_std_var, skip_sum = op.SkipSimplifiedLayerNormalization(
        input,
        skip,
        gamma,
        epsilon=epsilon,
        _outputs=4,
        _domain="com.microsoft",
    )
    return normalized, skip_sum


_skip_rms_rule = pattern.RewriteRule(_skip_rms_norm_pattern, _skip_rms_normalization)

skip_rms_normalization_rules = [_skip_rms_rule]
skip_rms_normalization_ruleset = pattern.RewriteRuleSet(skip_rms_normalization_rules)


def _skip_layer_norm_pattern(op, input, skip, gamma, beta, epsilon, stash_type):
    skip_sum = op.Add(input, skip)
    normalized = op.LayerNormalization(
        skip_sum,
        gamma,
        beta,
        axis=-1,
        epsilon=epsilon,
        stash_type=stash_type,
    )
    return normalized, skip_sum


def _skip_layer_normalization(op, input, skip, gamma, beta, epsilon, stash_type):
    if stash_type.value != 1:  # FLOAT type
        return None
    normalized, _mean, _inv_std_var, skip_sum = op.SkipLayerNormalization(
        input,
        skip,
        gamma,
        beta,
        epsilon=epsilon,
        _outputs=4,
        _domain="com.microsoft",
    )
    return normalized, skip_sum


# Fusion rule for Add + SkipLayerNormalization
def _skip_layer_norm_add_bias_pattern(op, input, skip, gamma, beta, bias, epsilon, stash_type):
    bias_sum = op.Add(input, bias)
    normalized, _mean, _inv_std_var, skip_sum = op.SkipLayerNormalization(
        bias_sum,
        skip,
        gamma,
        beta,
        epsilon=epsilon,
        _outputs=4,
        _domain="com.microsoft",
    )
    return normalized, skip_sum


def _skip_layer_normalization_add_bias(
    op, input, skip, gamma, beta, bias, epsilon, stash_type
):
    normalized, _mean, _inv_std_var, skip_sum = op.SkipLayerNormalization(
        input,
        skip,
        gamma,
        beta,
        bias,
        epsilon=epsilon,
        _outputs=4,
        _domain="com.microsoft",
    )
    return normalized, skip_sum


_skip_layer_rule = pattern.RewriteRule(
    _skip_layer_norm_pattern, _skip_layer_normalization, name="SkipLayerNorm"
)
_skip_layer_add_bias_rule = pattern.RewriteRule(
    _skip_layer_norm_add_bias_pattern,
    _skip_layer_normalization_add_bias,
    name="SkipLayerNormAddBias",
)


skip_layer_normalization_rules = [_skip_layer_rule, _skip_layer_add_bias_rule]
skip_layer_normalization_ruleset = pattern.RewriteRuleSet(skip_layer_normalization_rules)


fuse_skip_rms_normalization = _fusion_utils.apply_fusion_rules(skip_rms_normalization_ruleset)


fuse_skip_layer_normalization = _fusion_utils.apply_fusion_rules(
    skip_layer_normalization_ruleset
)
