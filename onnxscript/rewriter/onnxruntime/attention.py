# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnxscript.ir as ir
from onnxscript.rewriter import pattern

def sdpa_pattern(op, query, key_transposed, value, query_scale, key_scale, mask):
    scaled_query = op.Mul(query, query_scale)
    scaled_key_transposed = op.Mul(key_transposed, key_scale)
    attn_score = op.MatMul(scaled_query, scaled_key_transposed)
    masked_score = op.Add(attn_score, mask)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output

def sdpa(op, query, key_transposed, value, query_scale, key_scale, mask):
    # TODO
    # check if query_scale and key_scale are scalars == sqrt(sqrt(dimsize))
    return op.SDPA(query, key_transposed, value, query_scale, key_scale, mask, _domain="local")

def sdpa_pattern2(op, query, key_transposed, value, scale):
    attn_score = op.MatMul(query, key_transposed)
    masked_score = op.Div(attn_score, scale)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output

def sdpa2(op, query, key_transposed, value, scale):
    # TODO
    # check if scale == (sqrt(dimsize))
    return op.SDPA(query, key_transposed, value, scale, _domain="local")

def sdpa_pattern3(op, query, key_transposed, value, scale, mask):
    attn_score = op.MatMul(query, key_transposed)
    scaled_score = op.Div(attn_score, scale)
    masked_score = op.Add(scaled_score, mask)
    attn_weight = op.Softmax(masked_score, axis=-1)
    attn_output = op.MatMul(attn_weight, value)
    return attn_output

def sdpa3(op, query, key_transposed, value, scale, mask):
    # TODO
    # check if scale == (sqrt(dimsize))
    return op.SDPA(query, key_transposed, value, scale, mask, _domain="local")

rule = pattern.RewriteRule(sdpa_pattern, sdpa)
rule2 = pattern.RewriteRule(sdpa_pattern2, sdpa2)
rule3 = pattern.RewriteRule(sdpa_pattern3, sdpa3)

rules = pattern.RewriteRuleSet([rule, rule2, rule3])