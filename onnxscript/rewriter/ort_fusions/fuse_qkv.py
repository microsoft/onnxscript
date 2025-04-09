# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, pattern

Dim = Union[int, ir.SymbolicDim]


class PackedQKVForGQAFusion(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("PackedQKVForGQA", remove_nodes=False)

    def pattern(
        self,
        op,
        packed_qkv,
        past_key,
        past_value,
        seqlens_k,
        total_seq_length,
        cos,
        sin,
        q_num_heads,
        kv_num_heads,
        interleaved,
    ):
        """Pattern to detect sliced Q, K, V passed to GQA and replace with packed QKV."""

        # Slice packed QKV into query, key, and value
        query_BSD = op.Slice(packed_qkv, _allow_other_inputs=True, _outputs=["query_sliced"])
        key_BSDkv = op.Slice(packed_qkv, _allow_other_inputs=True, _outputs=["key_sliced"])
        value_BSDkv = op.Slice(packed_qkv, _allow_other_inputs=True, _outputs=["value_sliced"])

        # Pass sliced Q, K, V to GroupQueryAttention
        return op.GroupQueryAttention(
            query_BSD,
            key_BSDkv,
            value_BSDkv,
            past_key,
            past_value,
            seqlens_k,
            total_seq_length,
            cos,
            sin,
            # mask, # TODO: this is not a valid input for GQA
            num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            do_rotary=1,
            rotary_interleaved=interleaved,
            # skipped optional attributes: local_window_size, scale, smooth_softmax, softcap
            _domain="com.microsoft",
            _outputs=3,
        )

    def check(
        self,
        op,
        packed_qkv,
        query_sliced,
        key_sliced,
        value_sliced,
        **_,
    ):
        check_result = pattern.MatchResult()
        self.bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(self.bindings, val, dims)

        # Check packed_qkv shape (B, S, D)
        if no_match(packed_qkv, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {packed_qkv} does not match expected dimensions ['B', 'S', 'D']",
                packed_qkv,
            )

        # Check query, key, and value shapes (B, S, Dh)
        if no_match(query_sliced, ["B", "S", "Dq"]):
            return check_result.fail(
                f"Shape mismatch: {query_sliced} does not match expected dimensions ['B', 'S', 'Dq']",
                query_sliced,
            )
        if no_match(key_sliced, ["B", "S", "Dkv"]):
            return check_result.fail(
                f"Shape mismatch: {key_sliced} does not match expected dimensions ['B', 'S', 'Dkv']",
                key_sliced,
            )
        if no_match(value_sliced, ["B", "S", "Dkv"]):
            return check_result.fail(
                f"Shape mismatch: {value_sliced} does not match expected dimensions ['B', 'S', 'Dkv']",
                value_sliced,
            )

        # Ensure Dh = Dg + 2*Dkv
        D = self.bindings.get("D")
        Dq = self.bindings.get("Dq")
        Dkv = self.bindings.get("Dkv")

        if not isinstance(D, int) or not isinstance(Dq, int) or not isinstance(Dkv, int):
            return check_result.fail(
                "Could not determine the hidden sizes of query, key, and value.",
            )

        if Dq + (2 * Dkv) != D:  # type: ignore[operator]
            return check_result.fail(
                f"Hidden size of query, key and value do not add up to hidden size: {D} != {Dq} + (2 * {Dkv})",
            )

        return True

    def rewrite(
        self,
        op,
        packed_qkv,
        past_key,
        past_value,
        seqlens_k,
        total_seq_length,
        cos,
        sin,
        q_num_heads,
        kv_num_heads,
        interleaved,
        **_,
    ):
        """Rewrite the sliced Q, K, V into a packed QKV MatMul input for GQA."""

        # Pass packed QKV directly to GroupQueryAttention
        return op.GroupQueryAttention(
            packed_qkv,
            None,
            None,
            past_key,
            past_value,
            seqlens_k,
            total_seq_length,
            cos,
            sin,
            num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            do_rotary=1,
            rotary_interleaved=interleaved,
            _domain="com.microsoft",
            _outputs=3,
        )


# Define the fusion rule
packed_qkv_for_gqa_rule = PackedQKVForGQAFusion.rule()

# Add the rule to the GQA rewrite rule set
fuse_qkv_gqa_rules = pattern.RewriteRuleSet([packed_qkv_for_gqa_rule])

# Apply the fusion rules
fuse_qkv_gqa = _fusion_utils.apply_fusion_rules(fuse_qkv_gqa_rules)
