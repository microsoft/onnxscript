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

rule = pattern.RewriteRule(sdpa_pattern, sdpa)