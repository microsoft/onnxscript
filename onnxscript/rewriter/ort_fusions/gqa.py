# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

"""
GroupQueryAttention: This generalizes MHA by allowing the number of heads to be different
for query and key/value.

We use the following abbreviations for the dimensions:
B: Batch size
S: Sequence length (for current query/key/value)

Hkv: number of heads for key/value
G = number of groups
H: number of heads = G * Hkv

Dh: head size or embedding dimension per head
D: input embedding dimension (hidden size) = H * Dh
Dkv: key/value hidden size = Hkv * Dh

Skv: key/value sequence length (after concatenation of past and current key/value)

In the sequel, the suffix "_BHSDh" indicates that the tensor has the shape (B, H, S, Dh).
The suffix "BH_Skv_Dh" indicates that the tensor has the shape (B*H, Skv, Dh).
"""

Dim = Union[int, ir.SymbolicDim]


def _check_shape(bindings: dict[str, Dim], val: ir.Value, shape: Sequence[str]) -> bool:
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


class GroupQueryAttention(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("GQA")

    def pattern(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        mask,
        past_key,
        past_value,
        position_ids,
        cos,
        sin,
    ):
        # Reshape query from (B, S, D) to (B, S, H, D/H)
        query_BSHDh = op.Reshape(
            query_BSD,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["query_BSHDh"],
        )
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

        # Reshape key from (B, S, Dkv) to (B, S, Hkv, D/H)
        key_BSHkvDh = op.Reshape(
            key_BSDkv,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["key_BSHkvDh"],
        )
        # Transpose from (B, S, Hkv, D/H) to (B, Hkv, S, D/H)
        key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])

        # Reshape value from (B, S, Dkv) to (B, S, Hkv, D/H)
        value_BSHkvDh = op.Reshape(
            value_BSDkv,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["value_BSHkvDh"],
        )
        # Transpose from (B, S, Hkv, D/H) to (B, Hkv, S, D/H)
        value_BHkvSDh = op.Transpose(value_BSHkvDh, perm=[0, 2, 1, 3])

        position_ids_q = op.Unsqueeze(position_ids, [0])
        position_ids_k = op.Unsqueeze(position_ids, [0])

        query_BHSDh_rope = op.RotaryEmbedding(
            query_BHSDh, position_ids_q, cos, sin, _domain="com.microsoft"
        )
        key_BHkvSDh_rope = op.RotaryEmbedding(
            key_BHkvSDh, position_ids_k, cos, sin, _domain="com.microsoft"
        )

        # Concatenate past_key cache and current key, expand across heads
        # that share key/value and transpose to enable dot-product attention computation.

        key_seq_BHkvSkvDh = op.Concat(past_key, key_BHkvSDh_rope, axis=-2)
        key_seq_BHkv1SkvDh = op.Unsqueeze(key_seq_BHkvSkvDh, 2)
        key_seq_BHkvGSkvDh = op.Expand(key_seq_BHkv1SkvDh, _allow_other_inputs=True)
        key_seq_BHSkvDh = op.Reshape(
            key_seq_BHkvGSkvDh, _allow_other_inputs=True, _outputs=["key_seq_BHSkvDh"]
        )
        key_seq_BHDhSkv = op.Transpose(
            key_seq_BHSkvDh, _allow_other_inputs=True, _outputs=["key_seq_BHDhSkv"]
        )

        # Concatenate past_value cache and current value, expand across heads
        # that share key/value.
        value_seq_BHkvSkvDh = op.Concat(past_value, value_BHkvSDh, axis=-2)
        value_seq_BHkv1SkvDh = op.Unsqueeze(value_seq_BHkvSkvDh, 2)
        value_seq_BHkvGSkvDh = op.Expand(value_seq_BHkv1SkvDh, _allow_other_inputs=True)
        value_seq_BHSkvDh = op.Reshape(
            value_seq_BHkvGSkvDh, _allow_other_inputs=True, _outputs=["value_seq_BHSkvDh"]
        )

        attention_BHSDh = op.SDPA(
            query_BHSDh_rope,
            key_seq_BHDhSkv,
            value_seq_BHSkvDh,
            mask,
            _domain="ai.onnxruntime.fusion",
        )

        # Transpose attention back to (B, S, H, D/H)
        attention_BSHDh = op.Transpose(attention_BHSDh, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention_BSD = op.Reshape(
            attention_BSHDh, _allow_other_inputs=True, _outputs=["attention_BSD"]
        )
        return attention_BSD, key_seq_BHkvSkvDh, value_seq_BHkvSkvDh

    def check(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        mask,
        past_key,
        past_value,
        # query_BSHDh,
        # key_BSHkvDh,
        # value_BSHkvDh,
        **_,
    ):
        # bindings: dict[str, Dim] = {}

        # def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
        #     return not _check_shape(bindings, val, dims)

        # if no_match(query_BSD, ["B", "S", "D"]):
        #     return False
        # if no_match(key_BSDkv, ["B", "Skv", "D"]):
        #     return False
        # if no_match(value_BSDkv, ["B", "Skv", "D"]):
        #     return False

        # if no_match(past_key, ["B", "H", "Spast", "Dh"]):
        #     return False
        # if no_match(past_value, ["B", "H", "Spast", "Dv"]):
        #     return False
        # if no_match(query_BSHDh, ["B", "S", "H", "Dh"]):
        #     return False
        # if no_match(key_BSHkvDh, ["B", "S", "H", "Dh"]):
        #     return False
        # if no_match(value_BSHkvDh, ["B", "S", "H", "Dh"]):
        #     return False

        # TODO: mask shape check: ideally, it should be (1 or B, 1 or H, S, St)
        # But this also, unforunately, depends on ORT version.

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value

        return True

    def rewrite(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        mask,
        past_key,
        past_value,
        query_BSHDh,
        key_BSHkvDh,
        # key_BSHkvDh,
        # position_ids,
        # cos,
        # sin,
        **_,
    ):
        num_heads = _ir_utils.get_dim(query_BSHDh, 2)
        kv_num_heads = _ir_utils.get_dim(key_BSHkvDh, 2)
        if not isinstance(num_heads, int) or not isinstance(kv_num_heads, int):
            return None

        # # Switch to 3D RotaryEmbedding
        # # TODO: forward other attributes
        # query_BSD_rope = op.RotaryEmbedding(
        #     query_BSD, position_ids, cos, sin, _domain="com.microsoft"
        # )
        # key_BSD_rope = op.RotaryEmbedding(
        #     key_BSDkv, position_ids, cos, sin, _domain="com.microsoft"
        # )

        return op.GroupQueryAttention(
            query_BSD,
            key_BSDkv,
            value_BSDkv,
            past_key,
            past_value,
            # skipped optional inputs: seqlens_k, total_sequence_length, cos_cache, sin_cache
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            # skipped optional attributes: do_rotary, local_window_size, rotary_interleaved, scale, smooth_softmax, softcap
            _domain="com.microsoft",
            _outputs=3,
        )


_rule1 = GroupQueryAttention.rule()

gqa_rules = pattern.RewriteRuleSet([_rule1])


def fuse_gqa(model: ir.Model) -> int:
    count = gqa_rules.apply_to_model(model)
    print(f"GQA count: {count}")
    # remove_unused_nodes(model)
    return count
