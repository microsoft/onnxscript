# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript.rewriter import _ir_utils, pattern


def sdpa_pattern(op, query, key_transposed, value, query_scale, key_scale, mask):
    scaled_query = op.Mul(query, query_scale)
    scaled_key_transposed = op.Mul(key_transposed, key_scale)
    attn_score = op.MatMul(scaled_query, scaled_key_transposed)
    masked_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


def sdpa(op, query, key_transposed, value, query_scale, key_scale, mask):
    # Check if query_scale and key_scale are scalars == 1/sqrt(sqrt(dimsize))
    query_scale_value = _ir_utils.get_singleton_value(query_scale)
    key_scale_value = _ir_utils.get_singleton_value(key_scale)
    if not isinstance(query_scale_value, float) or not isinstance(key_scale_value, float):
        return None
    scaling_factor = query_scale_value * key_scale_value
    scaling_factor = 1.0 / (scaling_factor * scaling_factor)
    # If the dim_size is not statically known, we cannot check if the scale is correct:
    if query is None or query.shape is None or len(query.shape) < 2:
        return None
    dimsize = query.shape[-1]
    if not isinstance(dimsize, int) or not math.isclose(scaling_factor, dimsize, abs_tol=1e-3):
        return None
    return op.SDPA(query, key_transposed, value, mask, _domain="local")


def sdpa_pattern2(op, query, key_transposed, value, scale):
    attn_score = op.MatMul(query, key_transposed)
    masked_score = op.Div(attn_score, scale)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


def valid_post_scale(scale, query) -> bool:
    # Checks if scale == (sqrt(dimsize))
    scale_value = _ir_utils.get_singleton_value(scale)
    if not isinstance(scale_value, float):
        return False
    scaling_factor = scale_value * scale_value
    # If the dim_size is not statically known, we cannot check if the scale is correct:
    if query is None or query.shape is None or len(query.shape) < 2:
        return False
    dimsize = query.shape[-1]
    if not isinstance(dimsize, int) or not math.isclose(scaling_factor, dimsize, abs_tol=1e-3):
        return False
    return True


def sdpa2(op, query, key_transposed, value, scale):
    if not valid_post_scale(scale, query):
        return None
    return op.SDPA(query, key_transposed, value, scale, _domain="local")


def sdpa_pattern3(op, query, key_transposed, value, scale, mask):
    attn_score = op.MatMul(query, key_transposed)
    scaled_score = op.Div(attn_score, scale)
    masked_score = op.Add(scaled_score, mask)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output


def sdpa3(op, query, key_transposed, value, scale, mask):
    if not valid_post_scale(scale, query):
        return None
    return op.SDPA(query, key_transposed, value, scale, mask, _domain="local")


rule = pattern.RewriteRule(sdpa_pattern, sdpa)
rule2 = pattern.RewriteRule(sdpa_pattern2, sdpa2)
rule3 = pattern.RewriteRule(sdpa_pattern3, sdpa3)

sdpa_rules = pattern.RewriteRuleSet([rule, rule2, rule3])
