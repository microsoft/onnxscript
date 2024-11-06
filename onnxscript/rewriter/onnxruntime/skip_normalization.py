# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnxscript.ir as ir
from onnxscript.rewriter import pattern

def skip_norm_pattern(op, input, skip, gamma, epsilon, stash_type):
    skip_sum = op.Add(input, skip)
    normalized = op.SimplifiedLayerNormalization(
        skip_sum,
        gamma,
        axis=-1,
        epsilon=epsilon,
        stash_type=stash_type,
        _domain="com.microsoft")
    return normalized, skip_sum

def skip_normalization(op, input, skip, gamma, epsilon, stash_type):
    normalized, mean, inv_std_var, skip_sum = op.SkipSimplifiedLayerNormalization(
        input,
        skip,
        gamma,
        epsilon=epsilon,
        stash_type=stash_type,
        _domain="com.microsoft",
        _outputs=4
        )
    return normalized, skip_sum

rule = pattern.RewriteRule(skip_norm_pattern, skip_normalization, matcher=pattern.SimplePatternMatcher)