# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

Dim = Union[int, ir.SymbolicDim]


class PackedQKVForGQAFusion(pattern.RewriteRuleClassBase):
    def __init__(self, use_split=False):
        super().__init__(
            f"FusePackedQKVForGQAFrom{'Split' if use_split else 'Slices'}", remove_nodes=False
        )
        self._use_split = use_split

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
        start1,
        end1,
        start2,
        end2,
        start3,
        end3,
        split_sizes=None,
        split_axis=None,
    ):
        """Pattern to detect sliced Q, K, V passed to GQA and replace with packed QKV."""

        # Slice packed QKV into query, key, and value
        if self._use_split:
            query_BSD, key_BSDkv, value_BSDkv = op.Split(
                packed_qkv,
                split_sizes,
                axis=split_axis,
                _outputs=["query_sliced", "key_sliced", "value_sliced"],
            )
        else:
            query_BSD = op.Slice(packed_qkv, start1, end1, [2], [1], _outputs=["query_sliced"])
            key_BSDkv = op.Slice(packed_qkv, start2, end2, [2], [1], _outputs=["key_sliced"])
            value_BSDkv = op.Slice(
                packed_qkv, start3, end3, [2], [1], _outputs=["value_sliced"]
            )

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
            _allow_other_inputs=True,
            _domain="com.microsoft",
            _outputs=["attention", "present_key", "present_value"],
        )

    def check(
        self,
        op,
        packed_qkv,
        query_sliced,
        key_sliced,
        value_sliced,
        q_num_heads,
        kv_num_heads,
        start1=None,
        end1=None,
        start2=None,
        end2=None,
        start3=None,
        end3=None,
        split_sizes=None,
        split_axis=None,
        **_,
    ):
        check_result = pattern.MatchResult()
        self.bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils.check_shape_bool(self.bindings, val, dims)

        # Check that if x is being split into q, k, v correctly
        # based on hidden sizes
        if packed_qkv is None or packed_qkv.shape is None or len(packed_qkv.shape) != 3:
            return check_result.fail("packed_qkv is not a 3D tensor.", packed_qkv)
        hidden_size = packed_qkv.shape[2]
        if not isinstance(hidden_size, int):
            return check_result.fail("Hidden size is not an integer.", packed_qkv)
        if q_num_heads.is_ref() or kv_num_heads.is_ref():
            return check_result.fail(
                "Could not determine the number of heads for query, key and value.",
            )
        q_nh = q_num_heads.as_int()
        kv_nh = kv_num_heads.as_int()
        if q_nh <= 0 or kv_nh <= 0 or q_nh % kv_nh or hidden_size % (q_nh + 2 * kv_nh):
            return check_result.fail(
                "Packed width and positive query/KV head counts must agree."
            )
        head_size = hidden_size // (q_nh + (2 * kv_nh))
        if head_size <= 0:
            return check_result.fail("Packed head size must be positive.")
        q_hidden_size = head_size * q_nh
        kv_hidden_size = head_size * kv_nh
        if self._use_split:
            sizes = _ir_utils.get_numpy_value(split_sizes)
            if not (
                split_axis is not None
                and not split_axis.is_ref()
                and split_axis.as_int() in (2, -1)
                and split_sizes.dtype == ir.DataType.INT64
                and sizes is not None
                and sizes.ndim == 1
                and sizes.tolist() == [q_hidden_size, kv_hidden_size, kv_hidden_size]
            ):
                return check_result.fail(
                    "Split must partition the packed query, key, and value widths."
                )
        elif not (
            _ir_utils.is_singleton_value(start1, 0)
            and _ir_utils.is_singleton_value(end1, q_hidden_size)
            and _ir_utils.is_singleton_value(start2, q_hidden_size)
            and _ir_utils.is_singleton_value(end2, (q_hidden_size + kv_hidden_size))
            and _ir_utils.is_singleton_value(start3, (q_hidden_size + kv_hidden_size))
            and _ir_utils.is_singleton_value(end3, lambda x: x >= hidden_size)
        ):
            return check_result.fail(
                "packed_qkv is not being split into q, k, v correctly based on hidden sizes.",
                packed_qkv,
            )

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

        return check_result

    def rewrite(
        self,
        op,
        packed_qkv,
        attention,
        **_,
    ):
        """Rewrite the sliced Q, K, V into a packed QKV MatMul input for GQA."""

        original = attention.producer()
        inputs = list(original.inputs)
        inputs[:3] = [packed_qkv, None, None]
        return op.GroupQueryAttention(
            *inputs,
            **original.attributes,
            _domain="com.microsoft",
            _outputs=3,
        )


# Define the fusion rule
packed_qkv_for_gqa_rule = PackedQKVForGQAFusion.rule()

# Add the rule to the GQA rewrite rule set
fuse_qkv_gqa_rules = pattern.RewriteRuleSet(
    [packed_qkv_for_gqa_rule, PackedQKVForGQAFusion.rule(True)]
)

# Apply the fusion rules
fuse_qkv_gqa = _fusion_utils.apply_fusion_rules(fuse_qkv_gqa_rules)
