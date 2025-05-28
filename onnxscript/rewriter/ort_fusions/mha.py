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
    def __init__(
        self,
        name,
        *,
        double_transpose: bool,
        transpose_4d: bool,
        pre_scale_q: bool,
        is_rotary: bool,
        use_mask: bool,
        has_past_present: bool,
        is_cross_attention: bool,
    ):
        super().__init__(name)
        self._double_transpose = double_transpose
        self._transpose_4d = transpose_4d
        self._pre_scale_q = pre_scale_q
        self._is_rotary = is_rotary
        self._use_mask = use_mask
        self._has_past_present = has_past_present
        self._is_cross_attention = is_cross_attention

    def pattern(
        self,
        op,
        query_BSD,
        key,
        value,
        mask,
        past_key,
        past_value,
        position_ids,
        cos,
        sin,
        key_perm,
        q_scale,
    ):
        # First, query, key, and value are reshaped+transposed from (B, S, D) to (B, H, S, D/H)

        if self._pre_scale_q:
            query_BSD = op.Mul(query_BSD, q_scale)
        # Reshape from (B, S, D) to (B, S, H, D/H)
        query_BSHDh = op.Reshape(query_BSD, pattern.ANY_VALUE, _outputs=["query_BSHDh"])
        # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
        query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

        if not self._is_cross_attention:
            # Reshape from (B, S, D) to (B, S, H, D/H)
            key_BSHDh = op.Reshape(key, pattern.ANY_VALUE, _outputs=["key_BSHDh"])

            # Possible Transpose patterns for key:
            # This scenario optimizes the need for a double transpose
            # 1. (B, S, H, D/H) -> (B, H, D/H, S)
            # Patterns with double transpose of key
            # Double transpose should handle this optimization
            # 2. (B, S, H, D/H) -> (B, H, S, D/H) -> (B, H, D/H, S)
            # Patterns where key is reshaped to 3D, transposed and reshaped back to 4D
            # 3. (B, S, H, D/H) -> (B, H, S, D/H) -> R (B, S, D) -> (B, D, S) -> R (B, H, D/H, S)
            key_BHSDh = op.Transpose(key_BSHDh, perm=key_perm)

            # Reshape from (B, S, D) to (B, S, H, D/H)
            value_BSHDh = op.Reshape(value, pattern.ANY_VALUE, _outputs=["value_BSHDh"])
            # Transpose from (B, S, H, D/H) to (B, H, S, D/H)
            value_BHSDh = op.Transpose(value_BSHDh, perm=[0, 2, 1, 3])
        else:
            # For cross-attention, key and value are not reshaped
            key_BHSDh = key
            value_BHSDh = value

        if self._is_rotary:
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

            query_BHSDh_emb = op.RotaryEmbedding(
                query_BHSDh, position_ids_q, cos, sin, _domain="com.microsoft"
            )
            if not self._is_cross_attention:
                key_BHSDh_emb = op.RotaryEmbedding(
                    key_BHSDh, position_ids_k, cos, sin, _domain="com.microsoft"
                )
            else:
                key_BHSDh_emb = key_BHSDh
        else:
            # If rotary embedding is not used, we fuse with positional_embeddings
            query_BHSDh_emb = query_BHSDh
            key_BHSDh_emb = key_BHSDh

        # Concatenate past_key cache and current key, and transpose to enable
        # dot-product attention computation.
        if self._has_past_present:
            key_seq = op.Concat(past_key, key_BHSDh_emb, axis=-2)
        else:
            key_seq = key_BHSDh_emb

        # Concatenate past_value cache and current value
        if self._has_past_present:
            value_seq = op.Concat(past_value, value_BHSDh, axis=-2)
        else:
            value_seq = value_BHSDh

        # Key/value to be used for dot-product attention computation
        key_seq_to_sdpa = key_seq
        value_seq_to_sdpa = value_seq

        # Transpose last two axes of key_seq to compute dot-product via matmul.
        if self._double_transpose:
            if self._transpose_4d:
                key_seq_to_sdpa = op.Transpose(key_seq_to_sdpa, perm=[0, 1, 3, 2])
            else:
                # Transpose after converting to 3D
                key_seq_BH_Skv_Dh = op.Reshape(
                    key_seq_to_sdpa, pattern.ANY_VALUE, _outputs=["key_seq_BH_Skv_Dh"]
                )
                key_seq_BH_Dh_Skv = op.Transpose(key_seq_BH_Skv_Dh, perm=[0, 2, 1])
                key_seq_to_sdpa = op.Reshape(
                    key_seq_BH_Dh_Skv, pattern.ANY_VALUE, _outputs=["key_seq_B_H_Dh_Skv"]
                )

        # TODO: Remove use_mask once SDPA op is usable
        if self._use_mask:
            sdpa = op.SDPA(
                query_BHSDh_emb,
                key_seq_to_sdpa,
                value_seq_to_sdpa,
                mask,
                _domain="ai.onnxruntime.fusion",
            )
        else:
            sdpa = op.SDPA(
                query_BHSDh_emb,
                key_seq_to_sdpa,
                value_seq_to_sdpa,
                _domain="ai.onnxruntime.fusion",
            )

        # Transpose attention back to (B, S, H, D/H)
        attention_transposed = op.Transpose(sdpa, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention = op.Reshape(
            attention_transposed, pattern.ANY_VALUE, _outputs=["attention_reshaped"]
        )
        if self._has_past_present:
            return attention, key_seq, value_seq
        else:
            return attention

    def check(
        self,
        op,
        query_BSD,
        key,
        value,
        mask,
        past_key,
        past_value,
        key_perm,
        query_BSHDh,
        key_BSHDh=None,
        value_BSHDh=None,
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

        if no_match(query_BSHDh, ["B", "S", "H", "Dh"]):
            return check_result.fail(
                f"Shape mismatch: {query_BSHDh} does not match expected dimensions ['B', 'S', 'H', 'Dh']",
                query_BSHDh,
            )
        # If cross-attention, key/value shapes are 4D
        if self._is_cross_attention:
            if no_match(key, ["B", "H", "Skv", "Dh"]):
                return check_result.fail(
                    f"Shape mismatch: {key} does not match expected dimensions ['B', 'H', 'Skv', 'Dh']",
                    key,
                )
            if no_match(value, ["B", "H", "Skv", "Dv"]):
                return check_result.fail(
                    f"Shape mismatch: {value} does not match expected dimensions ['B', 'H', 'Skv', 'Dv']",
                    value,
                )
            # Ensure that no past_key/past_value is used in cross-attention
            if past_key is not None:
                return check_result.fail(
                    "past_key should be None in cross-attention.",
                    past_key,
                )
            if past_value is not None:
                return check_result.fail(
                    "past_value should be None in cross-attention.",
                    past_value,
                )
        else:
            if no_match(key, ["B", "Skv", "D"]):
                return check_result.fail(
                    f"Shape mismatch: {key} does not match expected dimensions ['B', 'Skv', 'D']",
                    query_BSD,
                )
            if no_match(value, ["B", "Skv", "D"]):
                return check_result.fail(
                    f"Shape mismatch: {value} does not match expected dimensions ['B', 'Skv', 'D']",
                    value,
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

        # mask (aka attention_bias) shape check:
        # ONNX's Attention op (named SDPA here) allows a mask broadcastable to (B, H, S, St)
        # ORT's contrib ops (MHA, Attention) allow a mask of shape (1 or B, 1 or H, S, St)
        # That is: broadcast allowed only for the first two dimensions. (Even that is not
        # supported by some earlier versions of ORT, which are not supported here.)
        if self._use_mask:
            if (mask_shape := mask.shape) is None:
                return check_result.fail(
                    "Mask shape cannot be determined.",
                    mask,
                )
            if mask_shape.rank() == 4:
                if no_match(mask, ["B_or_1", "H_or_1", "S_or_1", "St"]):
                    return check_result.fail(
                        f"Shape mismatch: {mask} does not match expected dimensions ['1 or B', '1 or H', '1 or S', 'St']",
                        mask,
                    )
                mask_dim_2 = bindings.get("S_or_1")
                if mask_dim_2 == bindings.get("S"):
                    self._use_mask_broadcast = False
                elif mask_dim_2 == 1:
                    self._use_mask_broadcast = True
                else:
                    return check_result.fail(
                        "Mask dimension 2 cannot be verified to be 1 or S"
                    )
            elif mask_shape.rank() == 2:
                if no_match(mask, ["S_or_1", "St"]):
                    return check_result.fail(
                        f"Shape mismatch: {mask} does not match expected dimensions ['1 or S', 'St']",
                        mask,
                    )
                self._use_mask_broadcast = True
            else:
                return check_result.fail(
                    f"Mask shape {mask_shape} is not supported. Expected 2D or 4D.",
                    mask,
                )
        else:
            self._use_mask_broadcast = False

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value
        return check_result

    def rewrite(
        self,
        op,
        query_BSD,
        key,
        value,
        mask,
        past_key,
        past_value,
        query_BSHDh,
        position_ids,
        cos,
        sin,
        q_scale=None,
        **_,
    ):
        scale = _ir_utils.get_singleton_value(q_scale)
        num_heads = _ir_utils.get_dim(query_BSHDh, 2)
        if not isinstance(num_heads, int):
            return None

        # TODO: forward other attributes

        if self._transpose_4d:
            zero_1d = op.Constant(value_ints=[0])
            position_ids = op.Unsqueeze(position_ids, zero_1d)

        if self._is_rotary:
            query_BSD_emb = op.RotaryEmbedding(
                query_BSD, position_ids, cos, sin, _domain="com.microsoft"
            )
            if not self._is_cross_attention:
                key_BSD_emb = op.RotaryEmbedding(
                    key, position_ids, cos, sin, _domain="com.microsoft"
                )
            else:
                key_BSD_emb = key
        elif self._is_cross_attention:
            query_BSD_emb = query_BSD
            # Must convert key/value from 4D to 3D for use in MHA
            key = op.Transpose(key, perm=[0, 2, 1, 3])
            key_BSD_emb = op.Reshape(key, op.Constant(value_ints=[0, 0, -1]))
            value = op.Transpose(value, perm=[0, 2, 1, 3])
            value = op.Reshape(value, op.Constant(value_ints=[0, 0, -1]))
        else:
            query_BSD_emb = query_BSD
            key_BSD_emb = key

        if self._use_mask_broadcast:
            one = op.Constant(value_ints=[1])
            S = op.Shape(query_BSD, start=1, end=2)
            shape_11S1 = op.Concat(one, one, S, one, axis=0)
            mask = op.Expand(mask, shape_11S1)

        num_outputs = 1 + (2 * self._has_past_present)
        return op.MultiHeadAttention(
            query_BSD_emb,
            key_BSD_emb,
            value,
            None,  # bias
            None,  # key padding mask
            mask,  # attention mask/bias
            past_key,
            past_value,
            num_heads=num_heads,
            scale=scale,
            _domain="com.microsoft",
            _outputs=num_outputs,
        )


def _make_rule_set(has_past_present: bool):
    parameter_combinations = [
        {
            "double_transpose": double_transpose,
            "transpose_4d": transpose_4d,
            "pre_scale_q": pre_scale_q,
            "is_rotary": is_rotary,
            "use_mask": use_mask,
            "has_past_present": has_past_present,
            "is_cross_attention": is_cross_attention,
        }
        for double_transpose in [False, True]
        for transpose_4d in (
            [False, True] if double_transpose else [False]
        )  # Only generate patterns when double_transpose is True
        for pre_scale_q in [True, False]
        for is_rotary in [False, True]
        for use_mask in [False, True]
        for is_cross_attention in ([False] if has_past_present else [False, True])
    ]

    # Dynamically create the rules
    mha_rules = pattern.RewriteRuleSet(
        [
            MultiHeadAttention.rule(
                f"MHA_{'4D' if params['transpose_4d'] else '3D'}_Transpose"
                f"{'_Twice' if params['double_transpose'] else ''}"
                f"{'_PreScaleQ' if params['pre_scale_q'] else ''}"
                f"{'_Rotary' if params['is_rotary'] else ''}"
                f"{'_Masked' if params['use_mask'] else ''}"
                f"{'_Past' if params['has_past_present'] else ''}"
                f"{'_CrossAttention' if params['is_cross_attention'] else ''}",
                **params,
            )
            for params in parameter_combinations
        ]
    )

    return mha_rules


mha_rules_no_past = _make_rule_set(has_past_present=False)
mha_rules_with_past = _make_rule_set(has_past_present=True)

# Try rules with past first, and then rules without past.
fuse_mha1 = _fusion_utils.apply_fusion_rules(mha_rules_with_past)
fuse_mha2 = _fusion_utils.apply_fusion_rules(mha_rules_no_past)
