# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import onnx_ir as ir

import onnxscript.onnx_types as _onnx_types
import onnxscript.rewriter._fusion_utils as _fusion_utils
from onnxscript.rewriter import _basics, _ir_utils, pattern

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


def _is_model_input(value: ir.Value, name: str, model: ir.Model) -> bool:
    return value in model.graph.inputs and value.name == name


def _causal_mask(
    op,
    input_ids,
    past_kv_cache,
    shape_B111,
    min_val,
    window_size,
    dtype,
):
    """Defines a pattern for a pure causal mask, with optional sliding window support."""
    seq_len = op.Shape(input_ids, end=2, start=1)
    seq_len_0D = op.Squeeze(seq_len)

    past_seq_len = op.Shape(past_kv_cache, end=3, start=2)
    past_seq_len_0D = op.Squeeze(past_seq_len)

    total_seq_len_0D = op.Add(past_seq_len_0D, seq_len_0D)
    total_seq_len = op.Reshape(total_seq_len_0D, [-1])

    current_range = op.Range(past_seq_len_0D, total_seq_len_0D, 1)
    mask_shape = op.Concat(seq_len, total_seq_len, axis=0)
    mask_all_min_expand = op.Expand(min_val, mask_shape)
    # The following Trilu is optional: not used in Phi models, but used in LLama.
    mask_all_min_trilu = op.Trilu(mask_all_min_expand, 1, upper=1)
    mask_all_min = pattern.OrValue([mask_all_min_expand, mask_all_min_trilu])
    total_range_as_row = op.Range(0, total_seq_len_0D, 1)
    current_range_as_column = op.Reshape(current_range, [-1, 1])

    non_causal = op.Greater(total_range_as_row, current_range_as_column)

    # sliding window support:
    current_range_minus_window = op.Sub(current_range_as_column, window_size)
    out_of_sliding_window = op.LessOrEqual(total_range_as_row, current_range_minus_window)
    non_causal_sliding_window = op.Or(non_causal, out_of_sliding_window)

    boolean_mask = pattern.OrValue([non_causal, non_causal_sliding_window])

    float_0_1_mask = op.Cast(boolean_mask, to=dtype)
    float_0_min_mask = op.Mul(mask_all_min, float_0_1_mask)
    mask_4d_11ST = op.Unsqueeze(float_0_min_mask, [0, 1])
    mask_4d_B1ST = op.Expand(mask_4d_11ST, shape_B111)

    return mask_4d_B1ST


class _CausalMaskPattern(pattern.PatternBase):
    def pattern(
        self,
        op,
        input_ids,
        past_kv_cache,
        shape_B111,
        min_val,
        window_size,
        dtype1,
        attn_mask_2d,
        dtype2,
    ):
        causal_mask = _causal_mask(
            op,
            input_ids,
            past_kv_cache,
            shape_B111,
            min_val,
            window_size,
            dtype1,
        )

        attn_mask_4d = op.Unsqueeze(attn_mask_2d, [1, 2])
        attn_mask_4d_cast = op.Cast(attn_mask_4d, to=dtype2)

        sum = op.Add(causal_mask, attn_mask_4d_cast)
        sum_fp32 = op.Cast(sum, to=ir.DataType.FLOAT)
        # The cast is optional, and may be absent if the sum is already in float32.
        sum_fp32 = pattern.OrValue([sum_fp32, sum])
        is_zero = op.Equal(sum_fp32, 0.0)
        result = op.Where(is_zero, min_val, causal_mask)
        return result

    def check(self, context, dtype1, dtype2, min_val, attn_mask_2d, sliding_window=None, **_):
        # Check that attn_mask_2d is the model input "attention_mask"
        if not _is_model_input(attn_mask_2d, "attention_mask", context.model):
            return pattern.MatchResult().fail("Invalid attention_mask input", attn_mask_2d)

        if dtype1.as_int() != dtype2.as_int():
            return pattern.MatchResult().fail("Dtype mismatch", [dtype1, dtype2])

        # Check that min_val is a constant and matches the expected minimum value for the dtype.
        min_value = _ir_utils.get_singleton_value(min_val)
        if min_value is None:
            return pattern.MatchResult().fail("Minval is not a constant.", min_val)
        expected_min_value = np.finfo(min_val.dtype.numpy()).min
        if min_value != expected_min_value:
            return pattern.MatchResult().fail(
                f"Expected min value {expected_min_value}, got {min_value}", min_val
            )

        # TODO(rama) Sliding window: not yet supported.
        if sliding_window:
            return pattern.MatchResult().fail(
                "Sliding window not yet supported", sliding_window
            )
        return True


_causal_mask_pattern = _CausalMaskPattern()


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
        position_ids,
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
            position_ids,
            cos,
            sin,
            _domain="com.microsoft",
            _outputs=["query_BHSDh_rope"],
        )
        key_BHkvSDh_rope = op.RotaryEmbedding(
            key_BHkvSDh,
            position_ids,
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
        context: _basics.MatchContext,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        query_BHSDh_rope,
        key_BHkvSDh_rope,
        query_BSHDh,
        key_BSHkvDh,
        mask,
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

        # Check mask:
        mask_node = mask.producer()
        if mask_node is None:
            return pattern.MatchResult().fail("Unhandled mask pattern", mask)
        mask_match_result = _causal_mask_pattern.match(
            context.model,
            context.graph_or_function,
            mask_node,
            check_nodes_are_removable=False,
        )
        if mask_match_result is None:
            return pattern.MatchResult().fail("Mask does not match causal mask pattern", mask)
        # TODO: handle sliding window support in mask

        return True

    def rewrite(
        self,
        op,
        query_BSD,
        key_BSDkv,
        value_BSDkv,
        past_key,
        past_value,
        position_ids,
        cos,
        sin,
        mask,
        **_,
    ):
        # Note that the following optimization is specific to current ORT GenAI attention-mask
        # usage. Specifically, it assumes that the model-input "attention_mask" is a 2D
        # mask with shape [batch_size, sequence_length], and that the mask is a 0/1 mask
        # that is used only to indicate the current tokens. Hence, the input attention_mask
        # is redundant as long as past-sequence-length and current-sequence-length can be
        # computed.

        # Construct seqlens_k and total_seq_length_int32 from position_ids
        # seqlens_k : int32[batch_size] indicates total_sequence-length-1 for each batch
        # position_ids: int64[batch_size, sequence_length] indicates the position of each token
        one_int32_0d = op.Constant(value=ir.tensor(1, dtype=ir.DataType.INT32))
        one_int64_1d = op.Constant(value=ir.tensor([1], dtype=ir.DataType.INT64))
        zero_int64_1d = op.Constant(value=ir.tensor([0], dtype=ir.DataType.INT64))
        seqlens_k_int64 = op.ReduceMax(position_ids, one_int64_1d, keepdims=0)
        seqlens_k = op.Cast(seqlens_k_int64, to=ir.DataType.INT32)
        max_seq_length = op.ReduceMax(seqlens_k, zero_int64_1d, keepdims=0)
        total_seq_length_int32 = op.Add(max_seq_length, one_int32_0d)
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
            num_heads=self.num_heads,
            kv_num_heads=self.kv_num_heads,
            do_rotary=1,
            rotary_interleaved=self._interleaved,
            # skipped optional attributes: local_window_size, scale, smooth_softmax, softcap
            _domain="com.microsoft",
            _outputs=3,
        )

class LongRoPeGQACausalMask(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("LongRoPeGQACausalMask", remove_nodes=False)
        self._mask_cache = {}
    
    def _get_mask_key(self, attention_mask):
        """
        Generate a unique key for the mask based on input_ids and past_kv_cache.
        This is used to cache the mask to avoid recomputation.
        """
        return (id(attention_mask))
    
    def compute_mask(self, op, attention_mask : _onnx_types.INT64['batch', 'seq_len']):
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
        seq_len = op.Shape(input_ids, end=2, start=1, _outputs=["seq_len"])
        seq_len_0D = op.Squeeze(seq_len, _outputs=["seq_len_0D"])
        past_seq_len = op.Shape(past_kv_cache_1, end=3, start=2, _outputs=["past_seq_len"])
        past_seq_len_0D = op.Squeeze(past_seq_len, _outputs=["past_seq_len_0D"])
        total_seq_len_0D = op.Add(past_seq_len_0D, seq_len_0D, _outputs=["total_seq_len_0D"])

        # All of the Add node's outputs
        current_range_A = op.Range(past_seq_len_0D, total_seq_len_0D, 1, _outputs=["current_range_A"])
        total_seq_len_A = op.Reshape(total_seq_len_0D, [-1], allowzero=0, _outputs=["total_seq_len_A"])
        current_range_B = op.Range(0, total_seq_len_0D, 1, _outputs=["current_range_B"])
        total_seq_len_B = op.Reshape(total_seq_len_0D, [-1], allowzero=0, _outputs=["total_seq_len_B"])
        total_seq_len_C = op.Reshape(total_seq_len_0D, [-1], allowzero=0, _outputs=["total_seq_len_C"])
            
        total_seq_len_final = op.Reshape(total_seq_len_0D, pattern.ANY_VALUE, allowzero=0, _outputs=["total_seq_len_final"])
    
        # EXPAND BRANCH A
        batch_size = op.Shape(past_kv_cache_2, end=1, start=0, _outputs=["batch_size"])
        mask_shape_A = op.Concat(batch_size, [1], seq_len, total_seq_len_A, axis=0, _outputs=["mask_shape_A"])
        mask_shape_A_abs = op.Abs(mask_shape_A, _outputs=["mask_shape_A_abs"])
        reshaped_range_A = op.Reshape(current_range_A, [1, 1, -1, 1], allowzero=1, _outputs=["reshaped_range_A"])
        mask_expanded_A = op.Expand(reshaped_range_A, mask_shape_A_abs, _outputs=["mask_expanded_A"])

        # EXPAND BRANCH B
        mask_shape_B = op.Concat(batch_size, [1], seq_len, total_seq_len_B, axis=0, _outputs=["mask_shape_B"])
        mask_shape_B_abs = op.Abs(mask_shape_B, _outputs=["mask_shape_B_abs"])
        reshaped_range_B = op.Reshape(current_range_B, [1, 1, 1, -1], allowzero=1, _outputs=["reshaped_range_B"])
        mask_expanded_B = op.Expand(reshaped_range_B, mask_shape_B_abs, _outputs=["mask_expanded_B"])
        
        # EXPAND BRANCH C
        mask_shape_C = op.Concat(batch_size, [1], seq_len, total_seq_len_C, axis=0, _outputs=["mask_shape_C"])
        mask_shape_C_abs = op.Abs(mask_shape_C, _outputs=["mask_shape_C_abs"])
        batch_size_squeezed = op.Squeeze(batch_size, _outputs=["batch_size_squeezed"])
        batch_range = op.Range(0, batch_size_squeezed, 1, _outputs=["batch_range"])
        reshaped_range_C = op.Reshape(batch_range, [-1, 1, 1, 1], allowzero=1, _outputs=["reshaped_range_C"])
        mask_expanded_C = op.Expand(reshaped_range_C, mask_shape_C_abs, _outputs=["mask_expanded_C"])

        # EXPAND A/B TO AND
        mask_expanded_A_sub = op.Sub(mask_expanded_A, 262144, _outputs=["mask_expanded_A_sub"])
        mask_A_B_greater = op.Greater(mask_expanded_B, mask_expanded_A_sub, _outputs=["mask_A_B_greater"])
        mask_A_B_greater_bitwise = op.And(True, mask_A_B_greater, _outputs=["mask_A_B_greater_bitwise"])
        mask_A_B_less = op.LessOrEqual(mask_expanded_B, mask_expanded_A, _outputs=["mask_A_B_less"])
        mask_A_B_combined = op.And(mask_A_B_greater_bitwise, mask_A_B_less, _outputs=["mask_A_B_combined"])
        mask_A_B_combined_bitwise = op.And(True, mask_A_B_combined, _outputs=["mask_A_B_combined_bitwise"])

        # EXPAND B/C TO AND
        unsqueezed_mask_expanded_B = op.Unsqueeze(mask_expanded_B, [-1], _outputs=["unsqueezed_mask_expanded_B"])
        unsqueezed_mask_expanded_C = op.Unsqueeze(mask_expanded_C, [-1], _outputs=["unsqueezed_mask_expanded_C"])
        mask_B_C_concat = op.Concat(unsqueezed_mask_expanded_C, unsqueezed_mask_expanded_B, axis=-1, _outputs=["mask_B_C_concat"])
        attention_mask_bool = op.Cast(attention_mask, to=ir.DataType.BOOL, _outputs=["attention_mask_bool"])
        mask_gatherND = op.GatherND(attention_mask_bool, mask_B_C_concat, batch_dims=0, _outputs=["mask_gatherND"])

        mask_A_B_C_combined = op.And(mask_A_B_combined_bitwise, mask_gatherND, _outputs=["mask_A_B_C_combined"])
        mask_A_B_C_negated = op.Not(mask_A_B_C_combined, _outputs=["mask_A_B_C_negated"])
        mask_A_B_C_fp32 = op.Cast(mask_A_B_C_negated, to=ir.DataType.FLOAT, _outputs=["mask_A_B_C_fp32"])
        mask_A_B_C_scaled = op.Mul(mask_A_B_C_fp32, pattern.ANY_VALUE)
        # Propagation to GQA
        mask_sliced = op.Slice(mask_A_B_C_scaled, [0], pattern.ANY_VALUE, [3], [1], _outputs=["mask_sliced"])

        #mask_where = op.Where(mask_sliced, pattern.ANY_VALUE, pattern.ANY_VALUE, _outputs=["mask_where"])

        return op.GQA(
            mask_sliced,
            pattern.ANY_VALUE,  # position_ids_k
            pattern.ANY_VALUE,  # position_ids_q  
            pattern.ANY_VALUE,  # query
            pattern.ANY_VALUE,  # key
            pattern.ANY_VALUE,  # value
            pattern.ANY_VALUE,  # past_key
            pattern.ANY_VALUE,  # past_value
            pattern.ANY_VALUE,  # seqlens_k (optional)
            pattern.ANY_VALUE,  # total_seq_length (optional)
            pattern.ANY_VALUE,  # cos
            pattern.ANY_VALUE,  # sin
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
_longrope_gqa_causal_mask_rule = LongRoPeGQACausalMask.rule()

gqa_rules = pattern.RewriteRuleSet([_basic_gqa_rule])
gqa_rules = pattern.RewriteRuleSet([_basic_gqa_rule, _longrope_gqa_causal_mask_rule])

fuse_gqa = _fusion_utils.apply_fusion_rules(gqa_rules)