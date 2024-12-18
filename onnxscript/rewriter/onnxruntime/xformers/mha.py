# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Iterable

import onnxscript.ir as ir
from onnxscript.rewriter import pattern

"""
The MultiHeadAttention pattern:

B: Batch size
S: Sequence length
D: input embedding dimension
H: number of heads
d_h: head size (usually, D = H * d_h)

thus, weights are usually of shape (D, D) and (D, D) and (D, D)

for each of Q, K, and V, we have the following pattern:
   MatMul (Input, W), producing output of shape (B, S, D)
   Reshape to produce a matrix of shape (B, S, H, d_h)
   Transpose middle two axes to produce a matrix of shape (B, H, S, d_h)

This is followed by a RotaryEmbedding pattern for Q and K

The last two axes of the key-embedding are then swapped (using a Reshape/Transpose/Reshape sequence)

The dot-product attention is then computed using SDPA
    
Finally, the output is transposed and reshaped back to (B, S, D) shape
"""


def _project_transpose_head(op, input, weight, reshape_var: str):
    """Applied to each of Q, K, and V."""
    # input_2d = op.Reshape(input, _allow_other_inputs=True, _allow_other_attributes=True)
    projected = op.MatMul(input, weight)
    # Reshape into 3D tensor (B, S, D)
    # reshaped_3d = op.Reshape(projected, _allow_other_inputs=True, _allow_other_attributes=True)
    # Reshape from (B, S, D) to (B, S, H, D/H)
    reshaped = op.Reshape(
        projected,
        _allow_other_inputs=True,
        _allow_other_attributes=True,
        _outputs=[reshape_var],
    )
    # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
    transposed = op.Transpose(reshaped, perm=[0, 2, 1, 3])
    return transposed


def _multi_head_attention_pattern(op, input, query_weight, key_weight, value_weight, cos, sin):
    query = _project_transpose_head(op, input, query_weight, "query_mm_reshaped")
    query_rope = op.RotaryEmbedding(query, cos, sin, _domain="local")
    key = _project_transpose_head(op, input, key_weight, "key_mm_reshaped")
    key_rope = op.RotaryEmbedding(key, cos, sin, _domain="local")
    # Transpose last two axes of key_rope to compute dot-product via matmul.
    key_reshaped = op.Reshape(key_rope, _allow_other_inputs=True, _outputs=["key_reshaped"])
    key_reshaped_transposed = op.Transpose(key_reshaped)
    key_transposed = op.Reshape(
        key_reshaped_transposed, _allow_other_inputs=True, _outputs=["key_transposed"]
    )
    value = _project_transpose_head(op, input, value_weight, "value_mm_reshaped")
    attention = op.SDPA(
        query_rope, key_transposed, value, _allow_other_inputs=True, _domain="local"
    )
    # Transpose back to (B, S, H, D/H)
    attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
    # Reshape back to (B, S, D)
    attention_reshaped = op.Reshape(
        attention_transposed, _allow_other_inputs=True, _outputs=["attention_reshaped"]
    )
    return attention_reshaped, key_rope, value


def _check_shape(bindings: dict[str, int], val: ir.Value, shape: Iterable[str]) -> bool:
    if val.shape is None:
        return False
    if val.shape.rank() != len(shape):
        return False
    for actual, expected in zip(val.shape, shape):
        if expected not in bindings:
            bindings[expected] = actual
        elif actual != bindings[expected]:
            return False
    return True


def _mha_validation(
    op,
    query_mm_reshaped,
    key_mm_reshaped,
    value_mm_reshaped,
    key_reshaped,
    key_transposed,
    attention_reshaped,
    **_,
):
    bindings: dict[str, int] = {}
    check = (
        _check_shape(bindings, query_mm_reshaped, ["B", "S", "H", "d_h"])
        and _check_shape(bindings, key_mm_reshaped, ["B", "S", "H", "d_h"])
        and _check_shape(bindings, value_mm_reshaped, ["B", "S", "H", "d_h"])
        and _check_shape(bindings, key_reshaped, ["B*H", "S", "d_h"])
        and _check_shape(bindings, key_transposed, ["B", "H", "d_h", "S"])
        and _check_shape(bindings, attention_reshaped, ["B", "S", "H*d_h"])
    )
    if not check:
        return False
    if bindings["B"] * bindings["H"] != bindings["B*H"]:
        return False
    if bindings["H"] * bindings["d_h"] != bindings["H*d_h"]:
        return False
    return True


def _multi_head_attention_pattern2(
    op, input, query_weight, key_weight, value_weight, cos, sin
):
    """Variation of first pattern with Reshape omitted."""
    query = _project_transpose_head(op, input, query_weight)
    query_rope = op.RotaryEmbedding(query, cos, sin, _domain="local")
    key = _project_transpose_head(op, input, key_weight)
    key_rope = op.RotaryEmbedding(key, cos, sin, _domain="local")
    # Transpose last two axes of key_rope to compute dot-product via matmul.
    # Reshape omitted here.
    key_transposed = op.Transpose(key_rope)
    # Reshape omitted here
    value = _project_transpose_head(op, input, value_weight)
    attention = op.SDPA(
        query_rope, key_transposed, value, _allow_other_inputs=True, _domain="local"
    )
    # Transpose back to (B, S, H, D/H)
    attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
    # Reshape back to (B, S, D)
    attention_reshaped = op.Reshape(attention_transposed, _allow_other_inputs=True)
    return attention_reshaped, key_rope, value


def _multi_head_attention(op, input, query_weight, key_weight, value_weight, cos, sin, **_):
    # TODO: other checks and concatenation of weights
    return op.MultiHeadAttention(
        input, query_weight, key_weight, value_weight, cos, sin, _domain="local", _outputs=3
    )


_rule1 = pattern.RewriteRule(
    _multi_head_attention_pattern, _multi_head_attention, _mha_validation
)

# TODO: _rule2 validation conditions
# _rule2 = pattern.RewriteRule(_multi_head_attention_pattern2, _multi_head_attention)

mha_rules = pattern.RewriteRuleSet([_rule1])
