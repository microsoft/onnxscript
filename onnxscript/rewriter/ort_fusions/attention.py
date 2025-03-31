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
    def __init__(self, name, *, has_input_bias: bool):
        super().__init__(name)
        # TODO: We can just pass bias to MultiHeadAttention
        # and let it handle the bias addition, once that pattern is added to MHA
        self._has_input_bias = has_input_bias

    def pattern(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        # mask_index,
        past,
        attention_bias,
        num_heads,
        scale,
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

        # Split past into past_key and past_value
        # past_key and past_value are of shape (B, H, S, D/H)
        '''
        past_key = op.Slice(
            past,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["past_key_sliced"],
        )
        past_value = op.Slice(
            past,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["past_value_sliced"],
        )
        '''

        # TODO: Pass other attributes
        attention = op.MultiHeadAttention(
            query_BSD,
            key_BSD,
            value_BSD,
            None,  # bias
            None,  # key_padding_mask
            attention_bias,
            # past_key,
            # past_value,
            num_heads=num_heads,
            scale=scale,
            _domain="com.microsoft",
            _outputs=3,
        )

        # Concat present_key and present_value to form present
        # present = op.Concat(present_key, present_value, axis=0)
        return attention#, present

    def check(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        query_mm_sliced,
        key_mm_sliced,
        value_mm_sliced,
        past_key_sliced,
        past_value_sliced,
        **_,
    ):
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _check_shape(bindings, val, dims)

        if no_match(input, ["B", "S", "D"]):
            return False
        if no_match(qkv_weight, ["D", "Dh"]):
            return False
        if no_match(qkv_bias, ["Dh"]):
            return False
        if no_match(query_mm_sliced, ["B", "S", "Dh_q"]):
            return False
        if no_match(key_mm_sliced, ["B", "S", "Dh_k"]):
            return False
        if no_match(value_mm_sliced, ["B", "S", "Dh_v"]):
            return False

        # Ensure Dh = Dh_q + Dh_k + Dh_v
        Dh = bindings.get("Dh")
        Dh_q = bindings.get("Dh_q")
        Dh_k = bindings.get("Dh_k")
        Dh_v = bindings.get("Dh_v")

        if Dh is None or Dh_q is None or Dh_k is None or Dh_v is None:
            return False  # Missing bindings, cannot verify

        if Dh != Dh_q + Dh_k + Dh_v:  # type: ignore[operator]
            return False  # Hidden size mismatch
        
        # Check that past is being split into two equal halves
        if no_match(past_key_sliced, ["B", "N", "S_past", "H"]):
            return False
        if no_match(past_value_sliced, ["B", "N", "S_past", "H"]):
            return False

        # TODO: Add mask check once mask is added to the pattern

        return True

    def rewrite(
        self,
        op,
        input,
        qkv_weight,
        qkv_bias,
        # mask_index,
        past,
        attention_bias,
        num_heads,
        scale,
        **_,
    ):
        return op.Attention(
            input,
            qkv_weight,
            qkv_bias,
            # mask_index
            # past,
            attention_bias,
            # past_sequence_length
            num_heads=num_heads,
            # qkv_hidden_sizes=qkv_hidden_sizes,
            scale=scale,
            _domain="com.microsoft",
            _outputs=2,
        )


attention_with_input_bias_rule = AttentionFusion.rule("attention_input_bias", has_input_bias=True)
attention_with_no_input_bias_rule = AttentionFusion.rule(
    "attention_no_input_bias", has_input_bias=False
)

attention_rules = pattern.RewriteRuleSet(
    [
        attention_with_input_bias_rule,
        attention_with_no_input_bias_rule,
    ]
)


def fuse_attention(model: ir.Model, *, debug: bool = False) -> int:
    count = attention_rules.apply_to_model(model)
    if debug and count == 0:
        tracer = pattern.MatchingTracer()
        attention_rules.apply_to_model(model, tracer=tracer)
        tracer.report()
    return count
