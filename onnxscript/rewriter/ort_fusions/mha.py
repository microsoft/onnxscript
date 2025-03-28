# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

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


class MultiHeadAttention(pattern.RewriteRuleClassBase):
    def __init__(self, name, *, transpose_4d: bool):
        super().__init__(name)
        self._transpose_4d = transpose_4d

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

        query_BHSDh_rope = op.RotaryEmbedding(
            query_BHSDh, position_ids_q, cos, sin, _domain="com.microsoft"
        )

        key_BHSDh_rope = op.RotaryEmbedding(
            key_BHSDh, position_ids_k, cos, sin, _domain="com.microsoft"
        )

        # Concatenate past_key cache and current key, and transpose to enable
        # dot-product attention computation.

        key_seq = op.Concat(past_key, key_BHSDh_rope, axis=-2)
        # Transpose last two axes of key_seq to compute dot-product via matmul.
        if self._transpose_4d:
            key_seq_B_H_Dh_Skv = op.Transpose(key_seq, perm=[0, 1, 3, 2])
        else:
            # Transpose after converting to 3D
            key_seq_BH_Skv_Dh = op.Reshape(
                key_seq, _allow_other_inputs=True, _outputs=["key_seq_BH_Skv_Dh"]
            )
            key_seq_BH_Dh_Skv = op.Transpose(key_seq_BH_Skv_Dh, perm=[0, 2, 1])
            key_seq_B_H_Dh_Skv = op.Reshape(
                key_seq_BH_Dh_Skv, _allow_other_inputs=True, _outputs=["key_seq_B_H_Dh_Skv"]
            )

        # Concatenate past_value cache and current value
        value_seq = op.Concat(past_value, value_BHSDh, axis=-2)

        attention = op.SDPA(
            query_BHSDh_rope,
            key_seq_B_H_Dh_Skv,
            value_seq,
            mask,
            _domain="ai.onnxruntime.fusion",
        )

        # Transpose attention back to (B, S, H, D/H)
        attention_transposed = op.Transpose(attention, perm=[0, 2, 1, 3])
        # Reshape back to (B, S, D)
        attention_reshaped = op.Reshape(
            attention_transposed, _allow_other_inputs=True, _outputs=["attention_reshaped"]
        )
        return attention_reshaped, key_seq, value_seq

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
        key_BSHDh,
        value_BSHDh,
        **_,
    ):
        bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _check_shape(bindings, val, dims)

        if no_match(query_BSD, ["B", "S", "D"]):
            return False
        if no_match(key_BSD, ["B", "Skv", "D"]):
            return False
        if no_match(value_BSD, ["B", "Skv", "D"]):
            return False

        if no_match(past_key, ["B", "H", "Spast", "Dh"]):
            return False
        if no_match(past_value, ["B", "H", "Spast", "Dv"]):
            return False
        if no_match(query_BSHDh, ["B", "S", "H", "Dh"]):
            return False
        if no_match(key_BSHDh, ["B", "S", "H", "Dh"]):
            return False
        if no_match(value_BSHDh, ["B", "S", "H", "Dh"]):
            return False
        # TODO: mask shape check: ideally, it should be (1 or B, 1 or H, S, St)
        # But this also, unforunately, depends on ORT version.

        # TODO: verify Reshapes:
        # eg.: verify bindings["B"] * bindings["H"] == bindings["B*H"]:
        # and bindings["H"] * bindings["Dh"] == bindings["H*Dh"]:
        # or check Reshape's shape-input value
        return True

    def rewrite(
        self,
        op,
        query_BSD,
        key_BSD,
        value_BSD,
        mask,
        past_key,
        past_value,
        key_BSHDh,
        position_ids,
        cos,
        sin,
        **_,
    ):
        num_heads = _ir_utils.get_dim(key_BSHDh, 2)
        if not isinstance(num_heads, int):
            return None

        # Switch to 3D RotaryEmbedding
        # TODO: forward other attributes

        if self._transpose_4d:
            zero_1d = op.Constant(value_ints=[0])
            position_ids = op.Unsqueeze(position_ids, zero_1d)
        query_BSD_rope = op.RotaryEmbedding(
            query_BSD, position_ids, cos, sin, _domain="com.microsoft"
        )
        key_BSD_rope = op.RotaryEmbedding(
            key_BSD, position_ids, cos, sin, _domain="com.microsoft"
        )

        return op.MultiHeadAttention(
            query_BSD_rope,
            key_BSD_rope,
            value_BSD,
            None,  # bias
            None,  # key padding mask
            mask,  # attention mask/bias
            past_key,
            past_value,
            num_heads=num_heads,
            _domain="com.microsoft",
            _outputs=3,
        )


_mha_4d_transpose = MultiHeadAttention.rule("MHA_4D_Transpose", transpose_4d=True)
_mha_3d_transpose = MultiHeadAttention.rule("MHA_3D_Transpose", transpose_4d=False)

mha_rules = pattern.RewriteRuleSet([_mha_4d_transpose, _mha_3d_transpose])


def fuse_mha(model: ir.Model, *, debug: bool = False) -> int:
    count = mha_rules.apply_to_model(model)
    if debug and count == 0:
        tracer = pattern.MatchingTracer()
        mha_rules.apply_to_model(model, tracer=tracer)
        tracer.report()
    return count
