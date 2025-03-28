# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.optimizer import remove_unused_nodes
from onnxscript.rewriter import pattern


class GroupQueryAttention(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, *, use_2d_matmul: bool):
        super().__init__(name, remove_nodes=False)
        self._use_2d_matmul = use_2d_matmul

    def _compute_packed_QKV(self, op, input, weight):
        if self._use_2d_matmul:
            # Convert batched input of shape (B, S, D) to 2D input (B*S, D)
            input = op.Reshape(input, _allow_other_inputs=True)
        projected = op.MatMul(input, weight)
        if self._use_2d_matmul:
            # Convert 2D output back to batched output of shape (B, S, D)
            projected = op.Reshape(projected, _allow_other_inputs=True)
        # Split combined QKV into Q, K, and V
        query_3d = op.Slice(projected, _allow_other_inputs=True)
        key_3d = op.Slice(projected, _allow_other_inputs=True)
        value_3d = op.Slice(projected, _allow_other_inputs=True)
        # Reshape from (B, S, D) to (B, S, H, D/H)
        query_4d = op.Reshape(
            query_3d,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["query_mm_reshaped"],
        )
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query = op.Transpose(query_4d, perm=[0, 2, 1, 3])
        key_4d = op.Reshape(
            key_3d,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["key_mm_reshaped"],
        )
        key = op.Transpose(key_4d, perm=[0, 2, 1, 3])
        value_4d = op.Reshape(
            value_3d,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["value_mm_reshaped"],
        )
        value = op.Transpose(value_4d, perm=[0, 2, 1, 3])

        return query, key, value

    def pattern(
        self,
        op,
        input,
        qkv_weight,
        mask,
        cos,
        sin,
        past_key,
        past_value,
        position_ids,
    ):
        query, key, value = self._compute_packed_QKV(op, input, qkv_weight)

        query_rope = op.RotaryEmbedding(query, position_ids, cos, sin, _domain="com.microsoft")

        key_rope = op.RotaryEmbedding(key, position_ids, cos, sin, _domain="com.microsoft")
        present_key = op.Concat(past_key, key_rope, axis=-2)
        # Transpose last two axes of present_key to compute dot-product via matmul.
        present_key = op.Transpose(present_key, perm=[0, 1, 3, 2])

        present_value = op.Concat(past_value, value, axis=-2)

        attention = op.SDPA(
            query_rope, present_key, present_value, mask, _domain="ai.onnxruntime.fusion"
        )
        # Transpose back to (B, S, H, D/H)
        attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention_reshaped = op.Reshape(
            attention_transposed, _allow_other_inputs=True, _outputs=["attention_reshaped"]
        )
        return attention_reshaped, present_key, present_value

    def check(
        self,
        op,
        # query_mm_reshaped,
        # key_mm_reshaped,
        # value_mm_reshaped,
        # key_reshaped,
        # key_transposed,
        # attention_reshaped,
        **_,
    ):
        # bindings: dict[str, int] = {}
        # status = (
        #     _check_shape(bindings, query_mm_reshaped, ["B", "S", "H", "d_h"])
        #     and _check_shape(bindings, key_mm_reshaped, ["B", "S", "H", "d_h"])
        #     and _check_shape(bindings, value_mm_reshaped, ["B", "S", "H", "d_h"])
        #     and _check_shape(bindings, key_reshaped, ["B*H", "KVS", "d_h"])
        #     and _check_shape(bindings, key_transposed, ["B", "H", "d_h", "KVS"])
        #     and _check_shape(bindings, attention_reshaped, ["B", "S", "H*d_h"])
        # )
        # if not status:
        #     return False
        # if bindings["B"] * bindings["H"] != bindings["B*H"]:
        #     return False
        # if bindings["H"] * bindings["d_h"] != bindings["H*d_h"]:
        #     return False
        return True

    def rewrite(
        self,
        op,
        input,
        qkv_weight,
        mask,
        cos,
        sin,
        past_key,
        past_value,
        position_ids,
        query_mm_reshaped,
        **_,
    ):
        num_heads = query_mm_reshaped.shape[2]
        qkv = op.MatMul(input, qkv_weight)
        return op.GroupQueryAttention(
            qkv,
            None,  # key
            None,  # value
            past_key,
            past_value,
            # seqlens_k,
            # total_sequence_length,
            cos,
            sin,
            num_heads=num_heads,
            _domain="com.microsoft",
            _outputs=3,
        )


_rule1 = GroupQueryAttention.rule("MHA_2dmm", use_2d_matmul=False)

gqa_rules = pattern.RewriteRuleSet([_rule1])


def fuse_gqa(model: ir.Model) -> int:
    count = gqa_rules.apply_to_model(model)
    print(f"GQA count: {count}")
    remove_unused_nodes(model)
    return count
