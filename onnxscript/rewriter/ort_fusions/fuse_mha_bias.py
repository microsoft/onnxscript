# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import numpy

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, pattern

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


class FuseBiasMHA(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name,
        *,
        q_no_bias: bool,
        k_no_bias: bool,
        v_no_bias: bool,
    ):
        super().__init__(name)
        self._q_no_bias = q_no_bias
        self._k_no_bias = k_no_bias
        self._v_no_bias = v_no_bias

    def pattern(
        self,
        op,
        query_matmul,
        key_matmul,
        value_matmul,
        q_bias,
        k_bias,
        v_bias,
        mask,
        past_key,
        past_value,
        num_heads,
    ):
        if not self._q_no_bias:
            query_BSD = op.Add(query_matmul, q_bias)
        else:
            query_BSD = query_matmul
        if not self._k_no_bias:
            key_BSD = op.Add(key_matmul, k_bias)
        else:
            key_BSD = key_matmul
        if not self._v_no_bias:
            value_BSD = op.Add(value_matmul, v_bias)
        else:
            value_BSD = value_matmul

        return op.MultiHeadAttention(
            query_BSD,
            key_BSD,
            value_BSD,
            None,  # bias
            None,  # key padding mask
            mask,  # attention mask/bias
            past_key,
            past_value,
            num_heads=num_heads,
            _domain="com.microsoft",
        )

    def check(
        self,
        op,
        query_matmul,
        key_matmul,
        value_matmul,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()

        self.bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(self.bindings, val, dims)

        if no_match(query_matmul, ["B", "S", "D"]):
            return check_result.fail(
                f"Shape mismatch: {query_matmul} does not match expected dimensions ['B', 'S', 'D']",
                query_matmul,
            )
        if no_match(key_matmul, ["B", "Skv", "Dk"]):
            return check_result.fail(
                f"Shape mismatch: {key_matmul} does not match expected dimensions ['B', 'Skv', 'Dk']",
                key_matmul,
            )
        if no_match(value_matmul, ["B", "Skv", "Dv"]):
            return check_result.fail(
                f"Shape mismatch: {value_matmul} does not match expected dimensions ['B', 'Skv', 'Dv']",
                value_matmul,
            )

        self.Dh_q = self.bindings.get("D")
        self.Dh_k = self.bindings.get("Dk")
        self.Dh_v = self.bindings.get("Dv")

        if (
            not isinstance(self.Dh_q, int)
            or not isinstance(self.Dh_k, int)
            or not isinstance(self.Dh_v, int)
        ):
            return check_result.fail(
                "Could not determine the hidden sizes of query, key, and value.",
            )

        return check_result

    def rewrite(
        self,
        op,
        query_matmul,
        key_matmul,
        value_matmul,
        q_bias,
        k_bias,
        v_bias,
        mask,
        past_key,
        past_value,
        num_heads,
        **_,
    ):
        if self._q_no_bias:
            q_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_q,), dtype=numpy.float32))
            )
        if self._k_no_bias:
            k_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_k,), dtype=numpy.float32))
            )
        if self._v_no_bias:
            v_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_v,), dtype=numpy.float32))
            )
        bias = op.Concat(q_bias, k_bias, v_bias, axis=0)
        return op.MultiHeadAttention(
            query_matmul,
            key_matmul,
            value_matmul,
            bias,
            None,
            mask,
            past_key,
            past_value,
            num_heads=num_heads,
            _domain="com.microsoft",
        )


parameter_combinations = [
    {
        "q_no_bias": q_no_bias,
        "k_no_bias": k_no_bias,
        "v_no_bias": v_no_bias,
    }
    for q_no_bias in [False, True]
    for k_no_bias in [False, True]
    for v_no_bias in [False, True]
]

# Dynamically create the rules
fuse_mha_bias_rules = pattern.RewriteRuleSet(
    [
        FuseBiasMHA.rule(
            f"MHABias{'_NoQBias' if params['q_no_bias'] else ''}"
            f"{'_NoKBias' if params['k_no_bias'] else ''}"
            f"{'_NoVBias' if params['v_no_bias'] else ''}",
            **params,
        )
        for params in parameter_combinations
    ]
)


fuse_mha_bias = _fusion_utils.apply_fusion_rules(fuse_mha_bias_rules)
