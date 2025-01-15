# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence

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

The dot-product attention is then computed using SDPA.
Finally, the output is transposed and reshaped back to (B, S, D) shape
"""


def _project_transpose_head(op, input, weight, reshape_var: str):
    """Applied to each of Q, K, and V."""
    projected = op.MatMul(input, weight)
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


def _multi_head_attention_pattern(
    op,
    input,
    query_weight,
    key_weight,
    value_weight,
    mask,
    cos,
    sin,
    past_key,
    past_value,
    position_ids,
):
    query = _project_transpose_head(op, input, query_weight, "query_mm_reshaped")
    query_rope = op.RotaryEmbedding(query, position_ids, cos, sin, _domain="com.microsoft")
    key = _project_transpose_head(op, input, key_weight, "key_mm_reshaped")
    key_rope = op.RotaryEmbedding(key, position_ids, cos, sin, _domain="com.microsoft")
    key_rope = op.Concat(past_key, key_rope, axis=-2)
    # Transpose last two axes of key_rope to compute dot-product via matmul.
    key_reshaped = op.Reshape(key_rope, _allow_other_inputs=True, _outputs=["key_reshaped"])
    key_reshaped_transposed = op.Transpose(key_reshaped, perm=[0, 2, 1])
    key_transposed = op.Reshape(
        key_reshaped_transposed, _allow_other_inputs=True, _outputs=["key_transposed"]
    )
    value = _project_transpose_head(op, input, value_weight, "value_mm_reshaped")
    value = op.Concat(past_value, value, axis=-2)
    attention = op.SDPA(
        query_rope, key_transposed, value, mask, _domain="ai.onnxruntime.fusion"
    )
    # Transpose back to (B, S, H, D/H)
    attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
    # Reshape back to (B, S, D)
    attention_reshaped = op.Reshape(
        attention_transposed, _allow_other_inputs=True, _outputs=["attention_reshaped"]
    )
    return attention_reshaped, key_rope, value


def _check_shape(bindings: dict[str, int], val: ir.Value, shape: Sequence[str]) -> bool:
    if val.shape is None:
        return False
    if val.shape.rank() != len(shape):
        return False
    for actual, expected in zip(val.shape, shape):
        if expected not in bindings:
            bindings[expected] = actual  # type: ignore[assignment]
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
        and _check_shape(bindings, key_mm_reshaped, ["B", "KVS", "H", "d_h"])
        and _check_shape(bindings, value_mm_reshaped, ["B", "KVS", "H", "d_h"])
        and _check_shape(bindings, key_reshaped, ["B*H", "TS", "d_h"])
        and _check_shape(bindings, key_transposed, ["B", "H", "d_h", "TS"])
        and _check_shape(bindings, attention_reshaped, ["B", "S", "H*d_h"])
    )
    if not check:
        return False
    if bindings["B"] * bindings["H"] != bindings["B*H"]:
        return False
    if bindings["H"] * bindings["d_h"] != bindings["H*d_h"]:
        return False
    return True


def _multi_head_attention(
    op,
    input,
    query_weight,
    key_weight,
    value_weight,
    mask,
    cos,
    sin,
    past_key,
    past_value,
    position_ids,
    query_mm_reshaped,
    **_,
):
    num_heads = query_mm_reshaped.shape[2]
    query = op.MatMul(input, query_weight)
    query_rope = op.RotaryEmbedding(query, position_ids, cos, sin, _domain="com.microsoft")
    key = op.MatMul(input, key_weight)
    key_rope = op.RotaryEmbedding(key, position_ids, cos, sin, _domain="com.microsoft")
    value = op.MatMul(input, value_weight)
    tiling_factor = op.Constant(value_ints=[1, num_heads, 1, 1])
    expanded_mask = op.Tile(mask, tiling_factor)
    return op.MultiHeadAttention(
        query_rope,
        key_rope,
        value,
        None,  # bias
        None,  # key padding mask
        expanded_mask,  # attention mask/bias
        past_key,
        past_value,
        num_heads=num_heads,
        _domain="com.microsoft",
        _outputs=3,
    )


_rule1 = pattern.RewriteRule(
    _multi_head_attention_pattern, _multi_head_attention, _mha_validation
)


mha_rules = pattern.RewriteRuleSet([_rule1])


def fuse_mha(model: ir.Model) -> int:
    count = mha_rules.apply_to_model(model)
    print(f"MHA count: {count}")
    return count
