# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

Dim = Union[int, ir.SymbolicDim]


# TODO: Maybe add this check to utilities


class AttentionFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name,
        *,
        has_past: bool,
        no_slice: bool,
    ):
        super().__init__(name)
        self._has_past = has_past
        self._no_slice = no_slice

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
        # scale,
        start1,
        end1,
        start2,
        end2,
        start3,
        end3,
        q_mul,
        k_mul,
        v_mul,
    ):
        if self._no_slice:
            query_BSD = op.MatMul(input, q_mul)
            key_BSD = op.MatMul(input, k_mul)
            value_BSD = op.MatMul(input, v_mul)
        else:
            projected = op.MatMul(input, qkv_weight, _outputs=["projected"])

            # Slice packed Matmul QKV into Q, K, and V
            # Q, K, and V are of shape (B, S, D)
            query_BSD = op.Slice(
                projected,
                start1,  # starts
                end1,  # ends
                [2],  # axes
                _outputs=["query_mm_sliced"],
            )
            key_BSD = op.Slice(
                projected,
                start2,  # starts
                end2,  # ends
                [2],  # axes
                _outputs=["key_mm_sliced"],
            )
            value_BSD = op.Slice(
                projected,
                start3,  # starts
                end3,  # ends
                [2],  # axes
                _outputs=["value_mm_sliced"],
            )

        # TODO: Add other attributes

        if self._has_past:
            # Split past into past_key and past_value
            # past_key and past_value are of shape (B, H, S, D/H)
            past_key = op.Slice(
                past,
                [0],  # starts
                [1],  # ends
                [0],  # axes
                _outputs=["past_key_sliced"],
            )
            past_key = op.Squeeze(past_key, [0])
            past_value = op.Slice(
                past,
                [1],  # starts
                [2],  # ends
                [0],  # axes
                _outputs=["past_value_sliced"],
            )
            past_value = op.Squeeze(past_value, [0])

            attention, present_key, present_value = op.MultiHeadAttention(
                query_BSD,
                key_BSD,
                value_BSD,
                qkv_bias,
                None,  # key_padding_mask
                attention_bias,
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
                qkv_bias,
                None,  # key_padding_mask
                attention_bias,
                None,  # past_key
                None,  # past_value
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
        projected=None,
        query_mm_sliced=None,
        key_mm_sliced=None,
        value_mm_sliced=None,
        start1=None,
        end1=None,
        start2=None,
        end2=None,
        start3=None,
        end3=None,
        q_mul=None,
        k_mul=None,
        v_mul=None,
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
        if not self._no_slice:
            # Ensure slicing is done correctly
            if projected is None or projected.shape is None or len(projected.shape) != 3:
                return check_result.fail("Input projection is not a 3D tensor.", projected)
            hidden_size = projected.shape[2]
            if not isinstance(hidden_size, int):
                return check_result.fail("Hidden size is not an integer.", projected)
            if not (
                _ir_utils.is_singleton_value(start1, 0)
                and _ir_utils.get_singleton_value(end1)
                == _ir_utils.get_singleton_value(start2)
                and _ir_utils.get_singleton_value(end2)
                == _ir_utils.get_singleton_value(start3)
                and _ir_utils.is_singleton_value(end3, lambda x: x >= hidden_size)
            ):
                return check_result.fail(
                    "Projected input is not being split into q, k, v correctly based on hidden sizes.",
                    projected,
                )

            if no_match(qkv_weight, ["D", "Dh"]):
                return check_result.fail(
                    f"Shape mismatch: {qkv_weight} does not match expected dimensions ['D', 'Dh']",
                    qkv_weight,
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
        else:
            if no_match(q_mul, ["D", "Dh_q"]):
                return check_result.fail(
                    f"Shape mismatch: {q_mul} does not match expected dimensions ['D', 'Dh_q']",
                    q_mul,
                )
            if no_match(k_mul, ["D", "Dh_k"]):
                return check_result.fail(
                    f"Shape mismatch: {k_mul} does not match expected dimensions ['D', 'Dh_k']",
                    k_mul,
                )
            if no_match(v_mul, ["D", "Dh_v"]):
                return check_result.fail(
                    f"Shape mismatch: {v_mul} does not match expected dimensions ['D', 'Dh_v']",
                    v_mul,
                )

        # Ensure Dh = Dh_q + Dh_k + Dh_v
        Dh = self.bindings.get("Dh")
        Dh_q = self.bindings.get("Dh_q")
        Dh_k = self.bindings.get("Dh_k")
        Dh_v = self.bindings.get("Dh_v")

        if not isinstance(Dh_q, int) or not isinstance(Dh_k, int) or not isinstance(Dh_v, int):
            return check_result.fail(
                "Could not determine the hidden sizes of query, key, and value.",
            )

        if not self._no_slice:
            if not isinstance(Dh, int):
                return check_result.fail(
                    "Could not determine the total hidden size of weight.",
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
        attention_bias,
        num_heads,
        # scale,
        q_mul=None,
        k_mul=None,
        v_mul=None,
        **_,
    ):
        # Use bindings to get the values of Dh_q, Dh_k, and Dh_v
        # and construct qkv_hidden_sizes
        Dh_q = self.bindings.get("Dh_q")
        Dh_k = self.bindings.get("Dh_k")
        Dh_v = self.bindings.get("Dh_v")
        qkv_hidden_sizes = [Dh_q, Dh_k, Dh_v]
        if self._no_slice:
            qkv_weight = op.Concat(q_mul, k_mul, v_mul, axis=1)

        if self._has_past:
            attention, present = op.Attention(
                input,
                qkv_weight,
                qkv_bias,
                None,  # mask_index
                past,
                attention_bias,
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
                None,  # mask_index
                None,  # past
                attention_bias,
                None,  # past_sequence_length
                num_heads=num_heads,
                qkv_hidden_sizes=qkv_hidden_sizes,
                # scale=scale,
                _domain="com.microsoft",
                _outputs=1,
            )


# Define all combinations of parameters
parameter_combinations = [
    {
        "name": f"attention_{'with_past_' if has_past else ''}{'no_slice' if no_slice else ''}".strip(
            "_"
        ),
        "has_past": has_past,
        "no_slice": no_slice,
    }
    for has_past in [False, True]
    for no_slice in [False, True]
]

# Dynamically create the rules
attention_rules = pattern.RewriteRuleSet(
    [
        AttentionFusion.rule(
            params["name"],
            has_past=params["has_past"],
            no_slice=params["no_slice"],
        )
        for params in parameter_combinations
    ]
)


fuse_attention = _fusion_utils.apply_fusion_rules(attention_rules)
