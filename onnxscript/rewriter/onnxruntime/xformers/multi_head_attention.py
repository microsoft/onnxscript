# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import pattern

"""
MultiHeadAttention:

D: input embedding dimension
H: number of heads
d_h: head size
usually, D = H * d_h

thus, weights are usually of shape (D, D) and (D, D) and (D, D)

for Q, K, V:
   MatMul (Input, W for Q, K, V) => B, S, D
   Reshape to B, S, 32, 64 (that is, B, S, H, d_h)
   Transpose to B, 32, S, 64 (that is, B, H, S, d_h)

Here, 32 is the number of heads and 64 is the head size

Embed Q and K

One of the embeddings (namely ) is also output of layer
and last two axes are transposed for SDPA

"""


def project_transpose_head(op, input, weight):
    projected = op.MatMul(input, weight)
    # Reshape from (B, S, D) to (B, S, H, D/H)
    reshaped = op.Reshape(projected, _allow_other_inputs=True, _allow_other_attributes=True)
    # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
    transposed = op.Transpose(reshaped, perm=[0, 2, 1, 3])
    return transposed


def multi_head_attention_pattern(op, input, query_weight, key_weight, value_weight, cos, sin):
    query = project_transpose_head(op, input, query_weight)
    query_rope = op.Embed(query, cos, sin, _domain="local")
    key = project_transpose_head(op, input, key_weight)
    key_rope = op.Embed(key, cos, sin, _domain="local")
    # Transpose last two axes of key_rope to compute dot-product via matmul.
    key_reshaped = op.Reshape(key_rope, _allow_other_inputs=True)
    key_reshaped_transposed = op.Transpose(key_reshaped)
    key_transposed = op.Reshape(key_reshaped_transposed, _allow_other_inputs=True)
    value = project_transpose_head(op, input, value_weight)
    attention = op.SDPA(
        query_rope, key_transposed, value, _allow_other_inputs=True, _domain="local"
    )
    # Transpose back to (B, S, H, D/H)
    attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
    # Reshape back to (B, S, D)
    attention_reshaped = op.Reshape(attention_transposed, _allow_other_inputs=True)
    return attention_reshaped, key_rope, value


def multi_head_attention_pattern2(op, input, query_weight, key_weight, value_weight, cos, sin):
    """Variation of first pattern with Reshape omitted."""
    query = project_transpose_head(op, input, query_weight)
    query_rope = op.Embed(query, cos, sin, _domain="local")
    key = project_transpose_head(op, input, key_weight)
    key_rope = op.Embed(key, cos, sin, _domain="local")
    # Transpose last two axes of key_rope to compute dot-product via matmul.
    # Reshape omitted here.
    key_transposed = op.Transpose(key_rope)
    # Reshape omitted here
    value = project_transpose_head(op, input, value_weight)
    attention = op.SDPA(
        query_rope, key_transposed, value, _allow_other_inputs=True, _domain="local"
    )
    # Transpose back to (B, S, H, D/H)
    attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
    # Reshape back to (B, S, D)
    attention_reshaped = op.Reshape(attention_transposed, _allow_other_inputs=True)
    return attention_reshaped, key_rope, value


def multi_head_attention(
    op,
    input,
    query_weight,
    key_weight,
    value_weight,
    cos,
    sin,
):
    # TODO: other checks and concatenation of weights
    return op.MultiHeadAttention(
        input, query_weight, key_weight, value_weight, cos, sin, _domain="local", _outputs=3
    )


rule = pattern.RewriteRule(multi_head_attention_pattern, multi_head_attention)
rule2 = pattern.RewriteRule(multi_head_attention_pattern2, multi_head_attention)

mha_rules = pattern.RewriteRuleSet([rule, rule2])
