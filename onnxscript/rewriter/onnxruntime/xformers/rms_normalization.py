# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern


# Pattern to match against
def _rms_norm_pattern(op, x, scale, epsilon, compute_dtype, target_dtype):
    x_cast = op.Cast(x, to=compute_dtype)
    x_square = op.Pow(x_cast, 2.0)
    mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
    mean_square_plus_epsilon = op.Add(mean_square, epsilon)
    rms = op.Sqrt(mean_square_plus_epsilon)
    reciprocal_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_cast, reciprocal_rms)
    normalized_cast = op.Cast(normalized, to=target_dtype)
    return op.Mul(scale, normalized_cast)

# Replacement
def _simplified_layer_norm(op, x, scale, epsilon, compute_dtype, target_dtype):
    epsilon_value = _ir_utils.get_singleton_value(epsilon)
    if not isinstance(epsilon_value, float):
        return None
    source_dtype = x.dtype
    if source_dtype is None or source_dtype != target_dtype.value:
        return None
    return op.SimplifiedLayerNormalization(
        x,
        scale,
        axis=-1,
        epsilon=epsilon_value,
        stash_type=compute_dtype.value,
    )

_rule = pattern.RewriteRule(_rms_norm_pattern, _simplified_layer_norm)


# Pattern to match against
def _rms_norm_pattern_no_cast(op, x, scale, epsilon):
    # x_cast = op.Cast(x, to=compute_dtype)
    x_cast = x
    x_square = op.Pow(x_cast, 2.0)
    mean_square = op.ReduceMean(x_square, [-1], keepdims=1, noop_with_empty_axes=0)
    mean_square_plus_epsilon = op.Add(mean_square, epsilon)
    rms = op.Sqrt(mean_square_plus_epsilon)
    reciprocal_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_cast, reciprocal_rms)
    # normalized_cast = op.Cast(normalized, to=target_dtype)
    normalized_cast = normalized
    return op.Mul(scale, normalized_cast)

# Replacement
def _simplified_layer_norm_no_cast(op, x, scale, epsilon):
    epsilon_value = _ir_utils.get_singleton_value(epsilon)
    if not isinstance(epsilon_value, float):
        return None
    source_dtype = x.dtype
    if source_dtype is None:
        return None
    return op.SimplifiedLayerNormalization(
        x,
        scale,
        axis=-1,
        epsilon=epsilon_value,
        stash_type=source_dtype.value,
    )

_rule_no_cast = pattern.RewriteRule(_rms_norm_pattern_no_cast, _simplified_layer_norm_no_cast)

rms_normalization_rules = pattern.RewriteRuleSet([_rule, _rule_no_cast])

def fuse_rms_normalization(model: ir.Model) -> None:
    count = rms_normalization_rules.apply_to_model(model)
    print(f"RMS Normalization count: {count}")
