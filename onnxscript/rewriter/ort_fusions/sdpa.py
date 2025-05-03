# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern


class SDPA(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name: str,
        *,
        use_mask: bool,
        pre_scale: bool,
        pre_scale_q: bool,
        use_mul: bool,
        has_3d_query: bool,
    ):
        super().__init__(name=name)
        self._use_mask = use_mask
        self._pre_scale = pre_scale
        # There are some patterns where only the query is scaled before the dot product
        # and essentially (query * qk_scale) * key is equivalent to (query * key) * qk_scale
        # TODO: Capture patterns where only the key is scaled before the dot product
        self._pre_scale_q = pre_scale_q
        self._use_mul = use_mul
        # Capture patterns where the query is reshaped from 3D to 4D
        # after scaling has been applied to query.
        self._has_3d_query = has_3d_query
        self._scale: float | None = None

    def pattern(
        self,
        op,
        query,
        key_transposed,
        value,
        mask,
        query_scale,
        key_scale,
        qk_scale,
        # Shape used for reshaping the query in patterns where query is reshaped
        # from 3D to 4D and scaling is applied before the reshaping.
        query_reshape,
    ):
        if self._pre_scale:
            # Some implementations scale the query and key before computing the dot product
            if self._use_mul:
                if self._pre_scale_q:
                    query = op.Mul(query, qk_scale)
                else:
                    query = op.Mul(query, query_scale)
                    key_transposed = op.Mul(key_transposed, key_scale)
            else:
                if self._pre_scale_q:
                    query = op.Div(query, qk_scale)
                else:
                    query = op.Div(query, query_scale)
                    key_transposed = op.Div(key_transposed, key_scale)

        # There might be patterns where the reshape and transpose are done
        # after the pre-scaling. If the inputs are 3D, we need to reshape them to 4D
        # and apply the approriate transposes to query.
        if self._has_3d_query and self._pre_scale_q:
            # Reshape and transpose 3D input of shape (B, S, D)
            # to 4D input of shape (B, N, S, H)
            queryBNSH = op.Reshape(query, query_reshape)
            query = op.Transpose(queryBNSH, perm=[0, 2, 1, 3])

        attn_score = op.MatMul(query, key_transposed)
        if not self._pre_scale:
            # Some implementations scale the dot product.
            if self._use_mul:
                attn_score = op.Mul(attn_score, qk_scale)
            else:
                attn_score = op.Div(attn_score, qk_scale)
        if self._use_mask:
            # Some implementations add a mask to the dot product.
            attn_score = op.Add(attn_score, mask)
        attn_weight = op.Softmax(attn_score, axis=-1)
        attn_output = op.MatMul(attn_weight, value)
        return attn_output

    def check(
        self, op, query, key_transposed, value, mask, query_scale, key_scale, qk_scale, **_
    ):
        check_result = pattern.MatchResult()
        # Check that the scaling factors match what SDPA implements:

        # We need to know the hidden size to check the scaling factors.
        if query is None or query.shape is None or len(query.shape) < 2:
            return check_result.fail(
                "Query shape is not known or has less than 2 dimensions.", query
            )
        hidden_size = query.shape[-1]
        if not isinstance(hidden_size, int):
            return check_result.fail("Hidden size is not an integer.")

        expected_scaling_factor = math.sqrt(hidden_size)
        if self._use_mul:
            expected_scaling_factor = 1.0 / expected_scaling_factor

        if self._pre_scale and not self._pre_scale_q:
            # Check if query_scale and key_scale are scalars == sqrt(expected_scaling_factor)
            # If they are scalars but != sqrt(expected_scaling_factor), a custom scale is being used.
            sqrt_scaling_factor = math.sqrt(expected_scaling_factor)
            # Calculate the scaling factor for query
            if (query_scale_value := _ir_utils.get_singleton_value(query_scale)) is None:
                return check_result.fail(
                    "Query scale is not a scalar.",
                    query_scale,
                )
            # Ensure the scaling factor for key is the same as for query
            if (key_scale_value := _ir_utils.get_singleton_value(key_scale)) is None:
                return check_result.fail(
                    "Key scale is not a scalar.",
                    key_scale,
                )
            if not math.isclose(query_scale_value, key_scale_value, rel_tol=1e-3):
                return check_result.fail(
                    "Query and key scales are not equal.",
                    query_scale,
                )
            if not math.isclose(query_scale_value, sqrt_scaling_factor, rel_tol=1e-3):
                self._scale = query_scale_value * query_scale_value
            else:
                # Pass no scaling factor to SDPA, SDPA will use the default scaling factor
                self._scale = None
        else:
            # Check if qk_scale is a scalar == expected_scaling_factor)
            # If it is a scalar but != sqrt(expected_scaling_factor), a custom scale is being used
            if (qk_scale_value := _ir_utils.get_singleton_value(qk_scale)) is None:
                return check_result.fail(
                    "QK scale is not a scalar.",
                    qk_scale,
                )
            if not math.isclose(qk_scale_value, expected_scaling_factor, rel_tol=1e-3):
                self._scale = qk_scale_value
            else:
                # Pass no scaling factor to SDPA, SDPA will use the default scaling factor
                self._scale = None

        # check ranks/shapes

        return check_result

    def rewrite(
        self,
        op,
        query,
        key_transposed,
        value,
        mask,
        query_scale,
        key_scale,
        qk_scale,
        query_reshape=None,
        **_,
    ):
        if self._pre_scale and self._pre_scale_q:
            if self._use_mul:
                query_mul = op.Mul(query, qk_scale)
            else:
                query_mul = op.Div(query, qk_scale)
            # Reshape and transpose 3D input of shape (B, S, D)
            # to 4D input of shape (B, N, S, H)
            if self._has_3d_query:
                queryBNSH = op.Reshape(query_mul, query_reshape)
                query = op.Transpose(queryBNSH, perm=[0, 2, 1, 3])
            else:
                query = query_mul

        sdpa_args = [query, key_transposed, value]
        if self._use_mask:
            sdpa_args.append(mask)
        return op.SDPA(*sdpa_args, scale=self._scale, _domain="ai.onnxruntime.fusion")


parameter_combinations = [
    {
        "name": f"sdpa_{'masked_' if use_mask else 'unmasked_'}{'pre_' if pre_scale else 'post_'}{'only_q_' if pre_scale_q else ''}{'mul' if use_mul else 'div'}{'_3d_query' if has_3d_query else ''}",
        "use_mask": use_mask,
        "pre_scale": pre_scale,
        "pre_scale_q": pre_scale_q,
        "use_mul": use_mul,
        "has_3d_query": has_3d_query,
    }
    for use_mask in [False, True]
    for pre_scale in [False, True]
    for pre_scale_q in [False, True]
    for use_mul in [False, True]
    for has_3d_query in [False, True]
]

# Dynamically create the rules
sdpa_rules = pattern.RewriteRuleSet(
    [
        SDPA.rule(
            params["name"],
            use_mask=params["use_mask"],
            pre_scale=params["pre_scale"],
            pre_scale_q=params["pre_scale_q"],
            use_mul=params["use_mul"],
            has_3d_query=params["has_3d_query"],
        )
        for params in parameter_combinations
    ]
)


fuse_sdpa = _fusion_utils.apply_fusion_rules(sdpa_rules)
