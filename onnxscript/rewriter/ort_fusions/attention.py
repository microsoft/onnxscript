# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, pattern

Dim = Union[int, ir.SymbolicDim]


# TODO: Maybe add this check to utilities


class AttentionFusion(pattern.RewriteRuleClassBase):
    def __init__(self, name, *, has_input_bias: bool, has_past: bool = False):
        super().__init__(name)
        # TODO: We can just pass bias to MultiHeadAttention
        # and let it handle the bias addition, once that pattern is added to MHA
        self._has_input_bias = has_input_bias
        self._has_past = has_past

    def pattern(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        # mask_index,
        past,
        # attention_bias,
        num_heads,
        # scale,
    ):
        projected = op.MatMul(input, qkv_weight)
        # Add bias if present
        if self._has_input_bias:
            projected = op.Add(projected, qkv_bias)

        # Slice packed Matmul QKV into Q, K, and V
        # Q, K, and V are of shape (B, S, D)
        query_BSD = op.Slice(
            projected,
            _allow_other_inputs=True,
            _outputs=["query_mm_sliced"],
        )
        key_BSD = op.Slice(
            projected,
            _allow_other_inputs=True,
            _outputs=["key_mm_sliced"],
        )
        value_BSD = op.Slice(
            projected,
            _allow_other_inputs=True,
            _outputs=["value_mm_sliced"],
        )

        # TODO: Add other attributes

        if self._has_past:
            # Split past into past_key and past_value
            # past_key and past_value are of shape (B, H, S, D/H)
            past_key = op.Slice(
                past,
                _allow_other_inputs=True,
                _outputs=["past_key_sliced"],
            )
            past_key = op.Squeeze(past_key, [0])
            past_value = op.Slice(
                past,
                _allow_other_inputs=True,
                _outputs=["past_value_sliced"],
            )
            past_value = op.Squeeze(past_value, [0])

            attention, present_key, present_value = op.MultiHeadAttention(
                query_BSD,
                key_BSD,
                value_BSD,
                None,  # bias
                None,  # key_padding_mask
                None,  # attention_bias,
                past_key,
                past_value,
                num_heads=num_heads,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=3,
            )
            # Concat present_key and present_value to form present
            present_key = op.Unsqueeze(present_key, [0])
            present_value = op.Unsqueeze(present_value, [0])
            present = op.Concat(present_key, present_value, axis=0)
            # Return present output first as it captures the complete pattern graph
            return present, attention
        else:
            attention = op.MultiHeadAttention(
                query_BSD,
                key_BSD,
                value_BSD,
                # bias
                # key_padding_mask
                # attention_bias,
                # past_key
                # past_value
                num_heads=num_heads,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=1,
            )
            return attention

    def check(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        query_mm_sliced,
        key_mm_sliced,
        value_mm_sliced,
        **_,
    ):
        check_result = pattern.MatchResult()
        self.bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(self.bindings, val, dims)

        if no_match(input, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {input} does not match expected dimensions ['B', 'S', 'D']",
                input,
            )
        if no_match(qkv_weight, ["D", "Dh"]):
            return check_result.fail(
                f"Shape mismatch: {qkv_weight} does not match expected dimensions ['D', 'Dh']",
                qkv_weight,
            )
        if no_match(qkv_bias, ["Dh"]):
            return check_result.fail(
                f"Shape mismatch: {qkv_bias} does not match expected dimensions ['Dh']",
                qkv_bias,
            )
        if no_match(query_mm_sliced, ["B", "S", "Dh_q"]):
            return check_result.fail(
                f"Shape mismatch: {query_mm_sliced} does not match expected dimensions ['B', 'S', 'Dh_q']",
                query_mm_sliced,
            )
        if no_match(key_mm_sliced, ["B", "S", "Dh_k"]):
            return check_result.fail(
                f"Shape mismatch: {key_mm_sliced} does not match expected dimensions ['B', 'S', 'Dh_k']",
                key_mm_sliced,
            )
        if no_match(value_mm_sliced, ["B", "S", "Dh_v"]):
            return check_result.fail(
                f"Shape mismatch: {value_mm_sliced} does not match expected dimensions ['B', 'S', 'Dh_v']",
                value_mm_sliced,
            )

        # Ensure Dh = Dh_q + Dh_k + Dh_v
        Dh = self.bindings.get("Dh")
        Dh_q = self.bindings.get("Dh_q")
        Dh_k = self.bindings.get("Dh_k")
        Dh_v = self.bindings.get("Dh_v")

        if (
            not isinstance(Dh, int)
            or not isinstance(Dh_q, int)
            or not isinstance(Dh_k, int)
            or not isinstance(Dh_v, int)
        ):
            return check_result.fail(
                "Could not determine the hidden sizes of query, key, and value.",
            )

        if Dh != Dh_q + Dh_k + Dh_v:  # type: ignore[operator]
            return check_result.fail(
                f"Hidden size of query, key and value do not add up to hidden size: {Dh} != {Dh_q} + {Dh_k} + {Dh_v}",
            )

        # TODO: Add mask check once mask is added to the pattern
        return check_result

    def rewrite(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        # mask_index,
        past,
        # attention_bias,
        num_heads,
        # scale,
        **_,
    ):
        # Use bindings to get the values of Dh_q, Dh_k, and Dh_v
        # and construct qkv_hidden_sizes
        Dh_q = self.bindings.get("Dh_q")
        Dh_k = self.bindings.get("Dh_k")
        Dh_v = self.bindings.get("Dh_v")
        qkv_hidden_sizes = [Dh_q, Dh_k, Dh_v]

        if self._has_past:
            attention, present = op.Attention(
                input,
                qkv_weight,
                qkv_bias,
                None,  # mask_index
                past,
                # attention_bias,
                # past_sequence_length
                num_heads=num_heads,
                qkv_hidden_sizes=qkv_hidden_sizes,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=2,
            )
            # Use same output ordering as in pattern
            return present, attention
        else:
            return op.Attention(
                input,
                qkv_weight,
                qkv_bias,
                # mask_index
                # past
                # attention_bias,
                # past_sequence_length
                num_heads=num_heads,
                qkv_hidden_sizes=qkv_hidden_sizes,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=1,
            )


attention = AttentionFusion.rule(
    "attention",
    has_input_bias=False,
    has_past=False,
)
attention_with_bias = AttentionFusion.rule(
    "attention_with_bias",
    has_input_bias=True,
    has_past=False,
)
attention_with_past = AttentionFusion.rule(
    "attention_with_past",
    has_input_bias=False,
    has_past=True,
)
attention_with_bias_and_past = AttentionFusion.rule(
    "attention_with_bias_and_past",
    has_input_bias=True,
    has_past=True,
)

attention_rules = pattern.RewriteRuleSet(
    [
        attention,
        attention_with_bias,
        attention_with_past,
        attention_with_bias_and_past,
    ]
)


fuse_attention = _fusion_utils.apply_fusion_rules(attention_rules)
