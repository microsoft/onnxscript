# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import pattern
from onnxscript.rewriter.onnxruntime.xformers.rms_normalization import rms_normalization_rules


def _skip_norm_pattern(op, input, skip, gamma, epsilon, stash_type):
    skip_sum = op.Add(input, skip)
    normalized = op.SimplifiedLayerNormalization(
        skip_sum,
        gamma,
        axis=-1,
        epsilon=epsilon,
        stash_type=stash_type,
    )
    return normalized, skip_sum


def _skip_normalization(op, input, skip, gamma, epsilon, stash_type):
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


_rule = pattern.RewriteRule(
    _skip_norm_pattern, _skip_normalization, matcher=pattern.SimplePatternMatcher
)

skip_normalization_rules = [_rule]
normalization_rules = rms_normalization_rules + skip_normalization_rules
normalization_ruleset = pattern.RewriteRuleSet(normalization_rules)


def fuse_normalization(model):
    count = normalization_ruleset.apply_to_model(model)
    print(f"Normalization count: {count}")
