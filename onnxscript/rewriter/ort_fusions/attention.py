# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import pattern

Dim = Union[int, ir.SymbolicDim]


# TODO: Maybe add this check to utilities
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
            _allow_other_attributes=True,
            _outputs=["query_mm_sliced"],
        )
        key_BSD = op.Slice(
            projected,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["key_mm_sliced"],
        )
        value_BSD = op.Slice(
            projected,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["value_mm_sliced"],
        )

        # TODO: Add other attributes

        if self._has_past:
            # Split past into past_key and past_value
            # past_key and past_value are of shape (B, H, S, D/H)
            past_key = op.Slice(
                past,
                _allow_other_inputs=True,
                _allow_other_attributes=True,
                _outputs=["past_key_sliced"],
            )
            past_key = op.Squeeze(past_key, [0])
            past_value = op.Slice(
                past,
                _allow_other_inputs=True,
                _allow_other_attributes=True,
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
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _check_shape(bindings, val, dims)

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
        Dh = bindings.get("Dh")
        Dh_q = bindings.get("Dh_q")
        Dh_k = bindings.get("Dh_k")
        Dh_v = bindings.get("Dh_v")

        if (
            not isinstance(Dh, int)
            or not isinstance(Dh_q, int)
            or not isinstance(Dh_k, int)
            or not isinstance(Dh_v, int)
        ):
            return False  # Missing bindings, cannot verify

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
                # qkv_hidden_sizes=qkv_hidden_sizes,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=2,
            )
            # TODO: Switch back order of outputs
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
                # qkv_hidden_sizes=qkv_hidden_sizes,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=1,
            )


attention_with_input_bias_rule = AttentionFusion.rule(
    "attention_input_bias",
    has_input_bias=True,
    has_past=False,
)
attention_with_no_input_bias_rule = AttentionFusion.rule(
    "attention_no_input_bias",
    has_input_bias=False,
    has_past=False,
)
attention_with_input_bias_with_past_rule = AttentionFusion.rule(
    "attention_input_bias_with_past",
    has_input_bias=True,
    has_past=True,
)
attention_with_no_input_bias_with_past_rule = AttentionFusion.rule(
    "attention_no_input_bias_with_past",
    has_input_bias=False,
    has_past=True,
)

attention_rules = pattern.RewriteRuleSet(
    [
        attention_with_input_bias_rule,
        attention_with_no_input_bias_rule,
        attention_with_input_bias_with_past_rule,
        attention_with_no_input_bias_with_past_rule,
    ]
)


def fuse_attention(model: ir.Model, *, debug: bool = False) -> int:
    count = attention_rules.apply_to_model(model)
    if debug and count == 0:
        tracer = pattern.MatchingTracer()
        attention_rules.apply_to_model(model, tracer=tracer)
        tracer.report()
    return count
