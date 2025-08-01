# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import onnx_ir as ir

import onnxscript.rewriter._fusion_utils as _fusion_utils
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

T: total sequence length (after concatenation of past and current key/value)
"""

Dim = Union[int, ir.SymbolicDim]


def causal_mask_pattern(op, input_ids, past_kv_cache, shape_B111):
    seq_len = op.Shape(input_ids, end=2, start=1)
    seq_len_0d = op.Squeeze(seq_len)

    past_seq_len = op.Shape(past_kv_cache, end=3, start=2)
    past_seq_len_0d = op.Squeeze(past_seq_len)

    total_seq_len_0d = op.Add(past_seq_len_0d, seq_len_0d)
    total_seq_len = op.Reshape(total_seq_len_0d, [-1])

    # The Phi modeling code generates the following +1 as the target-length, which seems
    # unnecessary in this context. But using it for pattern-matching against
    # generated onnx model.
    total_seq_len_plus_1_0d = op.Add(total_seq_len_0d, 1)
    total_seq_len_plus_1 = op.Reshape(total_seq_len_plus_1_0d, [-1])

    current_range = op.Range(past_seq_len_0d, total_seq_len_0d, 1)
    mask_shape = op.Concat(seq_len, total_seq_len_plus_1, axis=0)
    min_float32 = float(np.finfo(np.float32).min)
    mask_all_min = op.Expand(min_float32, mask_shape)
    total_range_as_row = op.Range(0, total_seq_len_plus_1_0d, 1)
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
        super().__init__("GQA", remove_nodes=False)

    def pattern(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        position_ids_q,
        position_ids_k,
        cos,
        sin,
        mask,
    ):
        # Reshape query from (B, S, D) to (B, S, H, D/H)
        query_BSHDh = op.Reshape(query_BSD, pattern.ANY_VALUE, _outputs=["query_BSHDh"])
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

        # Reshape key from (B, S, Dkv) to (B, S, Hkv, D/H)
        key_BSHkvDh = op.Reshape(key_BSDkv, pattern.ANY_VALUE, _outputs=["key_BSHkvDh"])
        # Transpose from (B, S, Hkv, D/H) to (B, Hkv, S, D/H)
        key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])

        # Reshape value from (B, S, Dkv) to (B, S, Hkv, D/H)
        value_BSHkvDh = op.Reshape(value_BSDkv, pattern.ANY_VALUE, _outputs=["value_BSHkvDh"])
        # Transpose from (B, S, Hkv, D/H) to (B, Hkv, S, D/H)
        value_BHkvSDh = op.Transpose(value_BSHkvDh, perm=[0, 2, 1, 3])

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
        # that share key/value.

        key_seq_BHkvTDh = op.Concat(past_key, key_BHkvSDh_rope, axis=-2)
        key_seq_BHkv1TDh = op.Unsqueeze(key_seq_BHkvTDh, 2)
        key_seq_BHkvGTDh = op.Expand(key_seq_BHkv1TDh, pattern.ANY_VALUE)
        key_seq_BHTDh = op.Reshape(
            key_seq_BHkvGTDh, pattern.ANY_VALUE, _outputs=["key_seq_BHTDh"]
        )

        # Concatenate past_value cache and current value, expand across heads
        # that share key/value.
        value_seq_BHkvTDh = op.Concat(past_value, value_BHkvSDh, axis=-2)
        value_seq_BHkv1TDh = op.Unsqueeze(value_seq_BHkvTDh, 2)
        value_seq_BHkvGTDh = op.Expand(value_seq_BHkv1TDh, pattern.ANY_VALUE)
        value_seq_BHTDh = op.Reshape(
            value_seq_BHkvGTDh, pattern.ANY_VALUE, _outputs=["value_seq_BHTDh"]
        )

        attention_BHSDh = op.SDPA(
            query_BHSDh_rope,
            key_seq_BHTDh,
            value_seq_BHTDh,
            mask,
            key_format="BHSd",
            _domain="ai.onnxruntime._fusion",
        )

        # Transpose attention back to (B, S, H, D/H)
        attention_BSHDh = op.Transpose(attention_BHSDh, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention_BSD = op.Reshape(
            attention_BSHDh, pattern.ANY_VALUE, _outputs=["attention_BSD"]
        )
        return attention_BSD, key_seq_BHkvTDh, value_seq_BHkvTDh

    def check(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        query_BHSDh_rope,
        key_BHkvSDh_rope,
        query_BSHDh,
        key_BSHkvDh,
        **_,
    ):
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(bindings, val, dims)

        if no_match(query_BSD, ["B", "S", "D"]):
            return False
        if no_match(key_BSDkv, ["B", "S", "Dkv"]):
            return False
        if no_match(value_BSDkv, ["B", "S", "Dkv"]):
            return False

        if no_match(past_key, ["B", "Hkv", "P", "Dh"]):
            return False
        if no_match(past_value, ["B", "Hkv", "P", "Dv"]):
            return False

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value

        result = pattern.MatchResult()
        num_heads = _ir_utils.get_dim(query_BSHDh, 2)
        kv_num_heads = _ir_utils.get_dim(key_BSHkvDh, 2)
        if not isinstance(num_heads, int):
            return result.fail("Unable to determine num_heads value", query_BSHDh)
        if not isinstance(kv_num_heads, int):
            return result.fail("Unable to determine kv_num_heads value", key_BSHkvDh)
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads

        # Rotary embedding attributes
        query_rotary_attributes = query_BHSDh_rope.producer().attributes
        key_rotary_attributes = key_BHkvSDh_rope.producer().attributes
        query_interleaved = query_rotary_attributes.get_int("interleaved", 0)
        key_interleaved = key_rotary_attributes.get_int("interleaved", 0)
        if query_interleaved != key_interleaved:
            return pattern.MatchResult().fail(
                "Rotary embedding interleaved attribute mismatch",
                [query_BHSDh_rope.producer(), key_BHkvSDh_rope.producer()],
            )
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
        position_ids_q,
        position_ids_k,
        cos,
        sin,
        mask,
        **_,
    ):
        return op.GQA(
            mask,
            position_ids_k,
            position_ids_q,
            query_BSD,
            key_BSDkv,
            value_BSDkv,
            past_key,
            past_value,
            None,  # seqlens_k,
            None,  # total_seq_length_int32,
            cos,
            sin,
            num_heads=self.num_heads,
            kv_num_heads=self.kv_num_heads,
            do_rotary=1,
            rotary_interleaved=self._interleaved,
            # skipped optional attributes: local_window_size, scale, smooth_softmax, softcap
            _domain="ai.onnxruntime._fusion",
            _outputs=3,
        )


class GQACausalMask(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("GQACausalMask", remove_nodes=False)

    def pattern(
        self,
        op,
        mask,
        input_ids,
        some_kv_cache,
        shape_B111,
        past_seq_length,
        total_seq_length,
    ):
        mask = causal_mask_pattern(op, input_ids, some_kv_cache, shape_B111)
        position_ids = op.Range(past_seq_length, total_seq_length, 1)
        position_ids_q = op.Unsqueeze(position_ids, [0])
        position_ids_k = op.Unsqueeze(position_ids, [0])
        return op.GQA(
            mask,
            position_ids_k,
            position_ids_q,
            _allow_other_inputs=True,
            _domain="ai.onnxruntime._fusion",
            _outputs=["attn_output", "key_seq", "value_seq"],
        )

    def rewrite(
        self,
        op,
        total_seq_length,
        attn_output,
        **_,
    ):
        # Construct total_seq_length_int32 and seqlens_k
        total_seq_length_int32 = op.Cast(total_seq_length, to=ir.DataType.INT32)
        one_0d = op.Constant(value_int=1)
        one_0d_int32 = op.Cast(one_0d, to=ir.DataType.INT32)
        seqlens_k_0d = op.Sub(total_seq_length_int32, one_0d_int32)
        zero_1d = op.Constant(value_int=0, dtype=ir.DataType.INT64, shape=[1])
        seqlens_k = op.Unsqueeze(seqlens_k_0d, zero_1d)

        gqa_node = attn_output.producer()
        assert len(gqa_node.inputs) == 12, (
            f"Expected 12 inputs for GQA node, got {len(gqa_node.inputs)}"
        )
        query, key, value, past_key, past_value = gqa_node.inputs[3:8]
        cos, sin = gqa_node.inputs[10:12]
        updated_inputs = [
            query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k,
            total_seq_length_int32,
            cos,
            sin,
        ]
        attributes = gqa_node.attributes
        return op.GroupQueryAttention(
            *updated_inputs, **attributes, _domain="com.microsoft", _outputs=3
        )

class LongRoPeGQACausalMask(pattern.RewriteRuleClassBase):
    """
    LongRoPeGQACausalMask is a specialized version of GQACausalMask that handles
    the LongRoPe GQA fusion. It computes the causal mask for Group Query Attention
    with LongRoPe (Long Range Rotary Position Embedding) and caches the mask to
    avoid recomputation at each layer.
    """
    def __init__(self):
        super().__init__("LongRoPeGQACausalMask", remove_nodes=False)
        self._mask_cache = {}

    def _get_mask_key(self, attention_mask):
        """
        Generate a unique key for the mask based on input_ids and past_kv_cache.
        This is used to cache the mask to avoid recomputation.
        """
        return (id(attention_mask))

    def compute_mask(self, op, attention_mask):
        """
        Computes the total_seq_length_int32 and seqlens_k_int32 based on the attention_mask,
        caching results to avoid recomputation at each layer.
        """
        mask_key = self._get_mask_key(attention_mask)

        if mask_key in self._mask_cache:
            total_seq_length_int32, seqlens_k_int32 = self._mask_cache[mask_key]

        else:
            # Construct total_seq_length_int32 and seqlens_k
            attention_shape = op.Shape(attention_mask, _outputs=["seq_len"])
            total_seq_length = op.Gather(attention_shape, op.Constant(value=ir.tensor(1, ir.DataType.INT64)), axis=0, _outputs=["total_seq_length"])
            reduced_attention = op.ReduceSum(attention_mask, op.Constant(value=ir.tensor([1], ir.DataType.INT64)), _outputs=["reduced_attention"])
            sub_reduced_attention = op.Sub(reduced_attention, op.Constant(value=ir.tensor([1], ir.DataType.INT64)), _outputs=["sub_reduced_attention"])
            total_seq_length_int32 = op.Cast(total_seq_length, to=ir.DataType.INT32, _outputs=["total_seq_length_int32"])
            seqlens_k_int32 = op.Cast(sub_reduced_attention, to=ir.DataType.INT32, _outputs=["seqlens_k_int32"])
            self._mask_cache[mask_key] = (total_seq_length_int32, seqlens_k_int32)

        return self._mask_cache[mask_key]


    def pattern(
        self,
        op,
        mask,
        input_ids,
        past_kv_cache_1,
        past_kv_cache_2,
        attention_mask,
        past_seq_length,
        total_seq_length,
    ):
        """
        Pattern for LongRoPe GQA Causal Mask.
        This pattern computes the causal mask for Group Query Attention with LongRoPe.
        It constructs the mask based on input_ids and past_kv_cache, and handles the
        expansion of the mask across the batch and sequence dimensions.
        """
        seq_len = op.Shape(input_ids, end=2, start=1, _outputs=["seq_len"])
        seq_len_0d = op.Squeeze(seq_len, _outputs=["seq_len_0d"])
        past_seq_len = op.Shape(past_kv_cache_1, end=3, start=2, _outputs=["past_seq_len"])
        past_seq_len_0d = op.Squeeze(past_seq_len, _outputs=["past_seq_len_0d"])
        total_seq_len_0d = op.Add(past_seq_len_0d, seq_len_0d, _outputs=["total_seq_len_0d"])

        # Create ranges for different dimensions
        kv_range = op.Range(past_seq_len_0d, total_seq_len_0d, 1, _outputs=["kv_range"])
        total_seq_len_for_kv = op.Reshape(total_seq_len_0d, [-1], allowzero=0, _outputs=["total_seq_len_for_kv"])
        query_range = op.Range(0, total_seq_len_0d, 1, _outputs=["query_range"])
        total_seq_len_for_query = op.Reshape(total_seq_len_0d, [-1], allowzero=0, _outputs=["total_seq_len_for_query"])
        total_seq_len_for_batch = op.Reshape(total_seq_len_0d, [-1], allowzero=0, _outputs=["total_seq_len_for_batch"])

        # BRANCH A: KV Range - Creates tensor with KV positions [1, 1, seq_len, 1]
        batch_size = op.Shape(past_kv_cache_2, end=1, start=0, _outputs=["batch_size"])
        kv_mask_shape = op.Concat(batch_size, [1], seq_len, total_seq_len_for_kv, axis=0, _outputs=["kv_mask_shape"])
        kv_mask_shape_abs = op.Abs(kv_mask_shape, _outputs=["kv_mask_shape_abs"])
        reshaped_kv_range = op.Reshape(kv_range, [1, 1, -1, 1], allowzero=1, _outputs=["reshaped_kv_range"])
        expanded_kv_range = op.Expand(reshaped_kv_range, kv_mask_shape_abs, _outputs=["expanded_kv_range"])

        # BRANCH B: Query Range - Creates tensor with query positions [1, 1, 1, total_seq_len]
        query_mask_shape = op.Concat(batch_size, [1], seq_len, total_seq_len_for_query, axis=0, _outputs=["query_mask_shape"])
        query_mask_shape_abs = op.Abs(query_mask_shape, _outputs=["query_mask_shape_abs"])
        reshaped_query_range = op.Reshape(query_range, [1, 1, 1, -1], allowzero=1, _outputs=["reshaped_query_range"])
        expanded_query_range = op.Expand(reshaped_query_range, query_mask_shape_abs, _outputs=["expanded_query_range"])

        # BRANCH C: Batch Range - Creates tensor with batch indices [batch_size, 1, 1, 1]
        batch_mask_shape = op.Concat(batch_size, [1], seq_len, total_seq_len_for_batch, axis=0, _outputs=["batch_mask_shape"])
        batch_mask_shape_abs = op.Abs(batch_mask_shape, _outputs=["batch_mask_shape_abs"])
        batch_size_squeezed = op.Squeeze(batch_size, _outputs=["batch_size_squeezed"])
        batch_range = op.Range(0, batch_size_squeezed, 1, _outputs=["batch_range"])
        reshaped_batch_range = op.Reshape(batch_range, [-1, 1, 1, 1], allowzero=1, _outputs=["reshaped_batch_range"])
        expanded_batch_range = op.Expand(reshaped_batch_range, batch_mask_shape_abs, _outputs=["expanded_batch_range"])

        # Combine KV/Query Ranges for Sliding Window Mask
        kv_range_offset = op.Sub(expanded_kv_range, 262144, _outputs=["kv_range_offset"])
        query_gt_kv_offset = op.Greater(expanded_query_range, kv_range_offset, _outputs=["query_gt_kv_offset"])
        query_gt_kv_offset_mask = op.And(True, query_gt_kv_offset, _outputs=["query_gt_kv_offset_mask"])
        query_le_kv = op.LessOrEqual(expanded_query_range, expanded_kv_range, _outputs=["query_le_kv"])
        sliding_window_mask = op.And(query_gt_kv_offset_mask, query_le_kv, _outputs=["sliding_window_mask"])
        sliding_window_mask_final = op.And(True, sliding_window_mask, _outputs=["sliding_window_mask_final"])

        # Combine Query/Batch Ranges for Attention Mask Lookup
        unsqueezed_query_range = op.Unsqueeze(expanded_query_range, [-1], _outputs=["unsqueezed_query_range"])
        unsqueezed_batch_range = op.Unsqueeze(expanded_batch_range, [-1], _outputs=["unsqueezed_batch_range"])
        batch_query_indices = op.Concat(unsqueezed_batch_range, unsqueezed_query_range, axis=-1, _outputs=["batch_query_indices"])
        attention_mask_bool = op.Cast(attention_mask, to=ir.DataType.BOOL, _outputs=["attention_mask_bool"])
        attention_lookup = op.GatherND(attention_mask_bool, batch_query_indices, batch_dims=0, _outputs=["attention_lookup"])

        # Final Mask Combination
        final_attention_mask = op.And(sliding_window_mask_final, attention_lookup, _outputs=["final_attention_mask"])
        inverted_mask = op.Not(final_attention_mask, _outputs=["inverted_mask"])
        mask_fp32 = op.Cast(inverted_mask, to=ir.DataType.FLOAT, _outputs=["mask_fp32"])
        scaled_mask = op.Mul(mask_fp32, pattern.ANY_VALUE)

        # Propagation to GQA
        sliced_mask = op.Slice(scaled_mask, [0], pattern.ANY_VALUE, [3], [1], _outputs=["sliced_mask"])

        gqa_input = pattern.OrValue([sliced_mask, scaled_mask])

        return op.GQA(
            gqa_input,
            _allow_other_inputs=True,
            _domain="ai.onnxruntime._fusion",
            _outputs=["attn_output", "key_seq", "value_seq"],
        )


    def rewrite(
        self,
        op,
        attention_mask,
        attn_output,
        **_,
    ):
        """
        Rewrite the GQA node with the new mask information.
        This method computes the total sequence length and seqlens_k based on the
        attention_mask and rewrites the GQA node to use these values.
        """
        # Compute total_seq_length_int32 and seqlens_k_int32
        total_seq_length_int32, seqlens_k_int32 = self.compute_mask(op, attention_mask)

        gqa_node = attn_output.producer()
        assert len(gqa_node.inputs) == 12, (
            f"Expected 12 inputs for GQA node, got {len(gqa_node.inputs)}"
        )
        query, key, value, past_key, past_value = gqa_node.inputs[3:8]
        cos, sin = gqa_node.inputs[10:12]
        updated_inputs = [
            query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k_int32,
            total_seq_length_int32,
            cos,
            sin,
        ]
        attributes = gqa_node.attributes
        return op.GroupQueryAttention(
            *updated_inputs, **attributes, _domain="com.microsoft", _outputs=3
        )

_basic_gqa_rule = GroupQueryAttention.rule()
_gqa_causal_mask_rule = GQACausalMask.rule()
_longrope_gqa_causal_mask_rule = LongRoPeGQACausalMask.rule()

gqa_rules = pattern.RewriteRuleSet([_basic_gqa_rule, _gqa_causal_mask_rule, _longrope_gqa_causal_mask_rule])

fuse_gqa = _fusion_utils.apply_fusion_rules(gqa_rules)
