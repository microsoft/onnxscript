# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

"""
The MultiHeadAttention pattern: generate an instance
   MHA (query, key, value, None, None, mask, past_key, past_value)
where query has shape (B, S, D), key has shape (B, Skv, D), and value has shape (B, Skv, Dv).
The next two inputs bias and key_padding_mask are None in this pattern. The mask (attention_bias)
must be  of shape (1 or B, 1 or H, S, St). past_key and past_value are of shape (B, H, Spast, Dh).

We use the following abbreviations for the dimensions:
B: Batch size
S: Sequence length
D: input embedding dimension
Dv: value hidden size (usually, Dv = D)
H: number of heads
Dh: head size or embedding dimension per head (usually, D = H * Dh)
Skv: key/value sequence length
St: total sequence length

In the sequel, the suffix "_BHSDh" indicates that the tensor has the shape (B, H, S, Dh).
The suffix "BH_Skv_Dh" indicates that the tensor has the shape (B*H, Skv, Dh).
"""

Dim = Union[int, ir.SymbolicDim]


class MultiHeadAttention(pattern.RewriteRuleClassBase):
    def __init__(self, name, *, transpose_4d: bool, is_rotary: bool, has_past_present: bool, is_sdpa_3d: bool, is_cross_attention: bool):
        super().__init__(name)
        self._transpose_4d = transpose_4d
        self._is_rotary = is_rotary
        self._has_past_present = has_past_present
        self._is_sdpa_3d = is_sdpa_3d
        self._is_cross_attention = is_cross_attention

    def pattern(
        self,
        op,
        query_BSD,
        key_BSD,
        value_BSD,
        mask,
        past_key,
        past_value,
        position_ids,
        cos,
        sin,
    ):
        # First, query, key, and value are reshaped+transposed from (B, S, D) to (B, H, S, D/H)

        # Reshape from (B, S, D) to (B, S, H, D/H)
        query_BSHDh = op.Reshape(
            query_BSD,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["query_BSHDh"],
        )
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

        # Reshape from (B, S, D) to (B, S, H, D/H)
        key_BSHDh = op.Reshape(
            key_BSD,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["key_BSHDh"],
        )
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        key_BHSDh = op.Transpose(key_BSHDh, perm=[0, 2, 1, 3])

        # Reshape from (B, S, D) to (B, S, H, D/H)
        value_BSHDh = op.Reshape(
            value_BSD,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["value_BSHDh"],
        )
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        value_BHSDh = op.Transpose(value_BSHDh, perm=[0, 2, 1, 3])

        # This is workaround for examples where there is a duplication of Unsqueeze op
        # to generate a 2D positions-ids from a 1D position-ids. This can be eliminated
        # if we have CSE-optimization to eliminate the duplicate Unsqueeze ops.
        # For now, same flag (transpose_4d) controls this variation. A different flag
        # can be added if we see instances that mix the two.
        if self._transpose_4d:
            position_ids_q = op.Unsqueeze(position_ids, [0])
            position_ids_k = op.Unsqueeze(position_ids, [0])
        else:
            position_ids_q = position_ids
            position_ids_k = position_ids

        if self._is_rotary:
            query_BHSDh_emb = op.RotaryEmbedding(
                query_BHSDh, position_ids_q, cos, sin, _domain="com.microsoft"
            )
            key_BHSDh_emb = op.RotaryEmbedding(
                key_BHSDh, position_ids_k, cos, sin, _domain="com.microsoft"
            )
        else:
            query_BHSDh_emb = query_BHSDh
            key_BHSDh_emb = key_BHSDh

        # Concatenate past_key cache and current key, and transpose to enable
        # dot-product attention computation.
        if self._has_past_present:
            key_seq = op.Concat(past_key, key_BHSDh_emb, axis=2)
        else:
            if self._is_cross_attention:
                key_seq = past_key
            else:
                key_seq = key_BHSDh_emb

        # Concatenate past_value cache and current value
        if self._has_past_present:
            value_seq = op.Concat(past_value, value_BHSDh, axis=2)
        else:
            if self._is_cross_attention:
                value_seq = past_value
            else:
                value_seq = value_BHSDh

        if self._is_sdpa_3d:
            query_BHSDh_emb = op.Reshape(query_BHSDh_emb, _allow_other_inputs=True)
            key_seq_to_sdpa = op.Reshape(key_seq, _allow_other_inputs=True)
            value_seq_to_sdpa = op.Reshape(value_seq, _allow_other_inputs=True)
        else:
            key_seq_to_sdpa = key_seq
            value_seq_to_sdpa = value_seq
        
        # Transpose last two axes of key_seq to compute dot-product via matmul.
        if self._transpose_4d:
            if self._is_sdpa_3d:
                key_seq_B_H_Dh_Skv = op.Transpose(key_seq_to_sdpa, perm=[0, 2, 1])
            else:
                key_seq_B_H_Dh_Skv = op.Transpose(key_seq_to_sdpa, perm=[0, 1, 3, 2])
        else:
            # Transpose after converting to 3D
            key_seq_BH_Skv_Dh = op.Reshape(
                key_seq_to_sdpa, _allow_other_inputs=True, _outputs=["key_seq_BH_Skv_Dh"]
            )
            key_seq_BH_Dh_Skv = op.Transpose(key_seq_BH_Skv_Dh, perm=[0, 2, 1])
            key_seq_B_H_Dh_Skv = op.Reshape(
                key_seq_BH_Dh_Skv, _allow_other_inputs=True, _outputs=["key_seq_B_H_Dh_Skv"]
            )
        
        attention = op.SDPA(
            query_BHSDh_emb,
            key_seq_B_H_Dh_Skv,
            value_seq_to_sdpa,
            _domain="ai.onnxruntime.fusion",
        )

        attention_reshaped = op.Reshape(attention, _allow_other_inputs=True)

        # Transpose attention back to (B, S, H, D/H)
        attention_transposed = op.Transpose(attention_reshaped, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention_reshaped = op.Reshape(
            attention_transposed, _allow_other_inputs=True, _outputs=["attention_reshaped"]
        )
        if self._has_past_present:
            return attention_reshaped, key_seq, value_seq
        else:
            return attention_reshaped

    def check(
        self,
        op,
        query_BSD,
        key_BSD,
        value_BSD,
        mask,
        past_key,
        past_value,
        query_BSHDh,
        #key_BSHDh,
        #value_BSHDh,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        bindings: dict[str, Dim] = {}
        
        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(bindings, val, dims)

        if no_match(query_BSD, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {query_BSD} does not match expected dimensions ['B', 'S', 'D']",
                query_BSD,
            )
        if not self._is_cross_attention:
            if no_match(key_BSD, ["B", "Skv", "D"]):
                return check_result.fail(
                    f"Shape mismatch: {key_BSD} does not match expected dimensions ['B', 'Skv', 'D']",
                    query_BSD,
                )
            if no_match(value_BSD, ["B", "Skv", "D"]):
                return check_result.fail(
                    f"Shape mismatch: {value_BSD} does not match expected dimensions ['B', 'Skv', 'D']",
                    value_BSD,
                )

        if self._has_past_present:
            if no_match(past_key, ["B", "H", "Spast", "Dh"]):
                return check_result.fail(
                    f"Shape mismatch: {past_key} does not match expected dimensions ['B', 'H', 'Spast', 'Dh']",
                    past_key,
                )
            if no_match(past_value, ["B", "H", "Spast", "Dv"]):
                return check_result.fail(
                    f"Shape mismatch: {past_value} does not match expected dimensions ['B', 'H', 'Spast', 'Dv']",
                    past_value,
                )
        if no_match(query_BSHDh, ["B", "S", "H", "Dh"]):
            return check_result.fail(
                f"Shape mismatch: {query_BSHDh} does not match expected dimensions ['B', 'S', 'H', 'Dh']",
                query_BSHDh,
            )
        '''
        if not self.is_cross_attention:
            if no_match(key_BSHDh, ["B", "S", "H", "Dh"]):
                return check_result.fail(
                    f"Shape mismatch: {key_BSHDh} does not match expected dimensions ['B', 'S', 'H', 'Dh']",
                    query_BSHDh,
                )
            if no_match(value_BSHDh, ["B", "S", "H", "Dh"]):
                return check_result.fail(
                    f"Shape mismatch: {value_BSHDh} does not match expected dimensions ['B', 'S', 'H', 'Dh']",
                    query_BSHDh,
                )
        '''
        # TODO: mask shape check: ideally, it should be (1 or B, 1 or H, S, St)
        # But this also, unforunately, depends on ORT version.

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value
        return check_result

    def rewrite(
        self,
        op,
        query_BSD,
        key_BSD,
        value_BSD,
        mask,
        past_key,
        past_value,
        query_BSHDh,
        position_ids,
        cos,
        sin,
        **_,
    ):
        num_heads = _ir_utils.get_dim(query_BSHDh, 2)
        if not isinstance(num_heads, int):
            return None

        # Switch to 3D RotaryEmbedding
        # TODO: forward other attributes

        if self._transpose_4d:
            zero_1d = op.Constant(value_ints=[0])
            position_ids = op.Unsqueeze(position_ids, zero_1d)
        
        if self._is_rotary:
            query_BSD_emb = op.RotaryEmbedding(
                query_BSD, position_ids, cos, sin, _domain="com.microsoft"
            )
            key_BSD_emb = op.RotaryEmbedding(
                key_BSD, position_ids, cos, sin, _domain="com.microsoft"
            )
        else:
            query_BSD_emb = query_BSD
            key_BSD_emb = key_BSD

        num_outputs = 1 + (2 * self._has_past_present)
        # Special case for cross-attention
        if self._is_cross_attention:
            return op.MultiHeadAttention(
            query_BSD_emb,
            past_key,
            past_value,
            None,  # bias
            None,  # key padding mask
            mask,  # attention mask/bias
            None,
            None,
            num_heads=num_heads,
            _domain="com.microsoft",
            _outputs=num_outputs,
        )

        return op.MultiHeadAttention(
            query_BSD_emb,
            key_BSD_emb,
            value_BSD,
            None,  # bias
            None,  # key padding mask
            mask,  # attention mask/bias
            past_key,
            past_value,
            num_heads=num_heads,
            _domain="com.microsoft",
            _outputs=num_outputs,
        )

parameter_combinations = [
    {"transpose_4d": transpose_4d, "is_rotary": is_rotary, "has_past_present": has_past_present, "is_sdpa_3d": is_sdpa_3d, "is_cross_attention": is_cross_attention}
    for transpose_4d in [True, False]
    for is_rotary in [True, False]
    for has_past_present in [True, False]
    for is_sdpa_3d in [True, False]
    for is_cross_attention in [False, True]
]

# Dynamically create the rules
mha_rules = pattern.RewriteRuleSet([
    MultiHeadAttention.rule(
        f"MHA_{'4D' if params['transpose_4d'] else '3D'}_Transpose"
        f"{'_Rotary' if params['is_rotary'] else ''}"
        f"{'_Past' if params['has_past_present'] else ''}"
        f"{'_SDPA_3D' if params['is_sdpa_3d'] else ''}"
        f"{'_CrossAttention' if params['is_cross_attention'] else ''}",
        **params
    )
    for params in parameter_combinations
])


fuse_mha = _fusion_utils.apply_fusion_rules(mha_rules)
