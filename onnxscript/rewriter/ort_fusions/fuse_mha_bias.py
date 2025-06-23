# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, Union

import numpy

import onnxscript.ir as ir
from onnxscript.rewriter import _fusion_utils, pattern

valid_float_types = [ir.DataType.FLOAT, ir.DataType.FLOAT16]

Dim = Union[int, ir.SymbolicDim]


class FuseBiasMHA(pattern.RewriteRuleClassBase):
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
        # scale,
    ):
        query_BSD = pattern.OrValue(
            [op.Add(query_matmul, q_bias), query_matmul],
            tag_var="has_q_bias",
            tag_values=[True, False],
        )
        key_BSD = pattern.OrValue(
            [op.Add(key_matmul, k_bias), key_matmul],
            tag_var="has_k_bias",
            tag_values=[True, False],
        )
        value_BSD = pattern.OrValue(
            [op.Add(value_matmul, v_bias), value_matmul],
            tag_var="has_v_bias",
            tag_values=[True, False],
        )

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
            # scale=scale,
            _domain="com.microsoft",
        )

    def check(
        self,
        context,
        query_matmul,
        key_matmul,
        value_matmul,
        has_q_bias,
        has_k_bias,
        has_v_bias,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()

        if not (has_q_bias or has_k_bias or has_v_bias):
            return check_result.fail("None of query, key, or value have a bias.")

        self.bindings: dict[str, Dim] = {}

        def no_match(val: ir.Value, dims: Sequence[str]) -> bool:
            return not _fusion_utils._check_shape(self.bindings, val, dims)

        if query_matmul.dtype not in valid_float_types:
            return check_result.fail("Query is not a float or float16 type.", query_matmul)
        if key_matmul.dtype not in valid_float_types:
            return check_result.fail("Key is not a float or float16 type.", key_matmul)
        if value_matmul.dtype not in valid_float_types:
            return check_result.fail("Value is not a float or float16 type.", value_matmul)

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
        # scale,
        **_,
    ):
        if q_bias is None:
            q_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_q,), dtype=query_matmul.dtype.numpy()))
            )
        if k_bias is None:
            k_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_k,), dtype=key_matmul.dtype.numpy()))
            )
        if v_bias is None:
            v_bias = op.Constant(
                value=ir.tensor(numpy.zeros((self.Dh_v,), dtype=value_matmul.dtype.numpy()))
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
            # scale=scale,
            _domain="com.microsoft",
        )


fuse_mha_bias_rules = pattern.RewriteRuleSet([FuseBiasMHA.rule()])


fuse_mha_bias = _fusion_utils.apply_fusion_rules(fuse_mha_bias_rules)
