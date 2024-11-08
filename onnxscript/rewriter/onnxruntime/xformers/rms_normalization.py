# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _ir_utils, pattern


# Pattern to match against
def rms_norm_pattern(op, x, scale, epsilon, compute_dtype, target_dtype):
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
def simplified_layer_norm(op, x, scale, epsilon, compute_dtype, target_dtype):
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
        _domain="com.microsoft",
    )


rms_normalization_rules = pattern.RewriteRule(rms_norm_pattern, simplified_layer_norm)
