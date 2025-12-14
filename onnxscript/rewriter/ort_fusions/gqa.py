# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import onnx_ir as ir

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
        # Qwen variant uses normalization of query/key before rotary embedding:
        # The normalization can happen before (eg., Qwen) or after the Transpose (eg., Gemma).
        query_BSHDh_normalized = op.SimplifiedLayerNormalization(
            query_BSHDh, pattern.ANY_VALUE, axis=-1, _outputs=["query_BSHDh_normalized"]
        )
        query_BSHDh = pattern.OrValue([query_BSHDh, query_BSHDh_normalized])

        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

        # Gemma variant uses normalization of query/key before rotary embedding:
        query_BHSDh_normalized = op.SimplifiedLayerNormalization(
            query_BHSDh, pattern.ANY_VALUE, axis=-1, _outputs=["query_BHSDh_normalized"]
        )
        query_BHSDh = pattern.OrValue([query_BHSDh, query_BHSDh_normalized])

        # Reshape key from (B, S, Dkv) to (B, S, Hkv, D/H)
        key_BSHkvDh = op.Reshape(key_BSDkv, pattern.ANY_VALUE, _outputs=["key_BSHkvDh"])
        key_BSHkvDh_normalized = op.SimplifiedLayerNormalization(
            key_BSHkvDh, pattern.ANY_VALUE, axis=-1, _outputs=["key_BSHkvDh_normalized"]
        )
        key_BSHkvDh = pattern.OrValue([key_BSHkvDh, key_BSHkvDh_normalized])

        # Transpose from (B, S, Hkv, D/H) to (B, Hkv, S, D/H)
        key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])

        # Gemma variant uses normalization of query/key before rotary embedding:
        key_BHkvSDh_normalized = op.SimplifiedLayerNormalization(
            key_BHkvSDh, pattern.ANY_VALUE, axis=-1, _outputs=["key_BHkvSDh_normalized"]
        )
        key_BHkvSDh = pattern.OrValue([key_BHkvSDh, key_BHkvSDh_normalized])

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
        # Concat with past_key is optional:
        key_seq_BHkvTDh = pattern.OrValue([key_seq_BHkvTDh, key_BHkvSDh_rope])
        key_seq_BHkv1TDh = op.Unsqueeze(key_seq_BHkvTDh, [2])
        key_seq_BHkvGTDh = op.Expand(key_seq_BHkv1TDh, pattern.ANY_VALUE)
        key_seq_BHTDh = op.Reshape(
            key_seq_BHkvGTDh, pattern.ANY_VALUE, _outputs=["key_seq_BHTDh"]
        )

        # Concatenate past_value cache and current value, expand across heads
        # that share key/value.
        value_seq_BHkvTDh = op.Concat(past_value, value_BHkvSDh, axis=-2)
        # Concat with past_value is optional:
        value_seq_BHkvTDh = pattern.OrValue([value_seq_BHkvTDh, value_BHkvSDh])
        value_seq_BHkv1TDh = op.Unsqueeze(value_seq_BHkvTDh, [2])
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
        query_BSHDh_normalized=None,
        query_BHSDh_normalized=None,
        key_BSHkvDh_normalized=None,
        key_BHkvSDh_normalized=None,
        **_,
    ):
        result = pattern.MatchResult()
        if query_BSHDh_normalized is not None and query_BHSDh_normalized is not None:
            return result.fail(
                "Query normalized twice",
                [query_BSHDh_normalized, query_BHSDh_normalized],
            )
        if key_BSHkvDh_normalized is not None and key_BHkvSDh_normalized is not None:
            return result.fail(
                "Key normalized twice",
                [key_BSHkvDh_normalized, key_BHkvSDh_normalized],
            )
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils.check_shape_bool(bindings, val, dims)

        if no_match(query_BSD, ["B", "S", "D"]):
            return False
        if no_match(key_BSDkv, ["B", "S", "Dkv"]):
            return False
        if no_match(value_BSDkv, ["B", "S", "Dkv"]):
            return False

        if past_key is not None and no_match(past_key, ["B", "Hkv", "P", "Dh"]):
            return False
        if past_value is not None and no_match(past_value, ["B", "Hkv", "P", "Dv"]):
            return False

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value

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
        query_BSHDh,
        key_BSHkvDh,
        query_BSHDh_normalized=None,
        query_BHSDh_normalized=None,
        key_BSHkvDh_normalized=None,
        key_BHkvSDh_normalized=None,
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

        normalized_query = query_BHSDh_normalized or query_BSHDh_normalized
        if normalized_query is not None:
            # We apply normalization without the transpose, which is fused into GQA
            norm_node = normalized_query.producer()
            norm_attrs = norm_node.attributes
            norm_scale = norm_node.inputs[1]
            query_BSHDh_normalized = op.SimplifiedLayerNormalization(
                query_BSHDh, norm_scale, **norm_attrs
            )
            reshape_BSHDh_to_BSD = op.Constant(value_ints=[0, 0, -1])
            query_BSD = op.Reshape(query_BSHDh_normalized, reshape_BSHDh_to_BSD)

        normalized_key = key_BHkvSDh_normalized or key_BSHkvDh_normalized
        if normalized_key is not None:
            # We apply normalization without the transpose, which is fused into GQA
            norm_node = normalized_key.producer()
            norm_attrs = norm_node.attributes
            norm_scale = norm_node.inputs[1]
            key_BSHkvDh_normalized = op.SimplifiedLayerNormalization(
                key_BSHkvDh, norm_scale, **norm_attrs
            )
            reshape_BSHkvDh_to_BSDkv = op.Constant(value_ints=[0, 0, -1])
            key_BSDkv = op.Reshape(key_BSHkvDh_normalized, reshape_BSHkvDh_to_BSDkv)

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


_basic_gqa_rule = GroupQueryAttention.rule()

gqa_rules = pattern.RewriteRuleSet([_basic_gqa_rule])

fuse_gqa = _fusion_utils.apply_fusion_rules(gqa_rules)
