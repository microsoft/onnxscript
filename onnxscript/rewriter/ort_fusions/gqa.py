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


def causal_mask_pattern(op, input_ids, past_kv_cache, shape_B111):
    seq_len = op.Shape(input_ids, end=2, start=1)
    seq_len_0D = op.Squeeze(seq_len)

    past_seq_len = op.Shape(past_kv_cache, end=3, start=2)
    past_seq_len_0D = op.Squeeze(past_seq_len)

    total_seq_len_0D = op.Add(past_seq_len_0D, seq_len_0D)
    total_seq_len = op.Reshape(total_seq_len_0D, [-1])

    # The Phi modeling code generates the following +1 as the target-length, which seems
    # unnecessary in this context. But using it for pattern-matching against
    # generated onnx model.
    total_seq_len_plus_1_0D = op.Add(total_seq_len_0D, 1)
    total_seq_len_plus_1 = op.Reshape(total_seq_len_plus_1_0D, [-1])

    current_range = op.Range(past_seq_len_0D, total_seq_len_0D, 1)
    mask_shape = op.Concat(seq_len, total_seq_len_plus_1, axis=0)
    mask_all_min = op.Expand(-3.4028235e38, mask_shape)
    total_range_as_row = op.Range(0, total_seq_len_plus_1_0D, 1)
    current_range_as_column = op.Reshape(current_range, [-1, 1])
    boolean_mask = op.Greater(total_range_as_row, current_range_as_column)
    float_0_1_mask = op.Cast(boolean_mask, to=1)
    float_0_min_mask = op.Mul(mask_all_min, float_0_1_mask)
    mask_4d = op.Unsqueeze(float_0_min_mask, [0, 1])
    mask_B1ST_plus = op.Expand(mask_4d, shape_B111)

    # Get rid of the extra +1 added above: total_seq_len is enough, no
    # need for total_seq_len+1.
    mask_B1ST = op.Slice(mask_B1ST_plus, [0], total_seq_len, [3], [1])
    return mask_B1ST


class GroupQueryAttention(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("GQA")
        self.remove_nodes = False

    def pattern(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        input_ids,
        past_seq_length,
        total_seq_length,
        cos,
        sin,
        some_kv_cache,
        shape_B111,
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

        position_ids = op.Range(past_seq_length, total_seq_length, 1)
        position_ids_q = op.Unsqueeze(position_ids, [0])
        position_ids_k = op.Unsqueeze(position_ids, [0])

        query_BHSDh_rope = op.RotaryEmbedding(
            query_BHSDh,
            position_ids_q,
            cos,
            sin,
            _domain="com.microsoft",
            _outputs=["query_BHSDh_rope"],
        )
        key_BHkvSDh_rope = op.RotaryEmbedding(
            key_BHkvSDh,
            position_ids_k,
            cos,
            sin,
            _domain="com.microsoft",
            _outputs=["key_BHkvSDh_rope"],
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

        mask = causal_mask_pattern(op, input_ids, some_kv_cache, shape_B111)

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
        query_BHSDh_rope,
        key_BHkvSDh_rope,
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
        # TODO: check that mask is causal. Latest ORT is adding support for
        # non-causal masks, but not yet for all EPs.

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value

        # Rotary embedding attributes
        query_rotary_attributes = query_BHSDh_rope.producer().attributes
        key_rotary_attributes = key_BHkvSDh_rope.producer().attributes
        query_interleaved = query_rotary_attributes.get("interleaved", 0)
        key_interleaved = key_rotary_attributes.get("interleaved", 0)
        if query_interleaved != key_interleaved:
            return False
        self._interleaved = query_interleaved

        return True

    def rewrite(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        query_BSHDh,
        key_BSHkvDh,
        total_seq_length,
        cos,
        sin,
        **_,
    ):
        num_heads = _ir_utils.get_dim(query_BSHDh, 2)
        kv_num_heads = _ir_utils.get_dim(key_BSHkvDh, 2)
        if not isinstance(num_heads, int) or not isinstance(kv_num_heads, int):
            return None

        total_seq_length_int32 = op.Cast(total_seq_length, to=ir.DataType.INT32)
        one_0D = op.Constant(value_int=1)
        one_0D_int32 = op.Cast(one_0D, to=ir.DataType.INT32)
        seqlens_k_0D = op.Sub(total_seq_length_int32, one_0D_int32)
        zero_1D = op.Constant(value_int=0, dtype=ir.DataType.INT64, shape=[1])
        seqlens_k = op.Unsqueeze(seqlens_k_0D, zero_1D)

        return op.GroupQueryAttention(
            query_BSD,
            key_BSDkv,
            value_BSDkv,
            past_key,
            past_value,
            seqlens_k,
            total_seq_length_int32,
            cos,
            sin,
            # mask, # TODO: this is not a valid input for GQA
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            do_rotary=1,
            rotary_interleaved=self._interleaved.value,
            # skipped optional attributes: local_window_size, scale, smooth_softmax, softcap
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
