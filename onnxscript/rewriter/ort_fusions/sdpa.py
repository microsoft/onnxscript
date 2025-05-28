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
        # Some implementations scale the query and key before computing the dot product
        query = pattern.OrValue([
            op.Mul(query, query_scale),
            op.Div(query, query_scale),
            query,
        ], tag_var="query_scaling", tag_values=["Mul", "Div", "None"])
        key_transposed = pattern.OrValue([
            op.Mul(key_transposed, key_scale),
            op.Div(key_transposed, key_scale),
            key_transposed,
        ], tag_var="key_scaling", tag_values=["Mul", "Div", "None"])


        attn_score = op.MatMul(query, key_transposed)

        # Some implementations scale the dot product.
        attn_score = pattern.OrValues ([
            op.Mul(attn_score, qk_scale),
            op.Div(attn_score, qk_scale),
            attn_score,
        ], tag_var="qk_scaling", tag_values=["Mul", "Div", "None"])

        # Some implementations add a mask to the dot product.
        masked_attn_score = op.Add(attn_score, mask)
        attn_score = pattern.OrValue([masked_attn_score, attn_score], tag_var="has_mask", tag_values=[True, False])

        attn_weight = op.Softmax(attn_score, axis=-1)
        attn_output = op.MatMul(attn_weight, value)
        return attn_output

    def check(
        self, op, query, key_transposed, value, mask, query_scaling, query_scale, key_scaling, key_scale, qk_scale, **_
    ):
        check_result = pattern.MatchResult()
        
        if query_scaling == "None":
            query_scale_value = 1.0
        elif query_scaling == "Mul":
            if (query_scale_value := _ir_utils.get_singleton_value(query_scale, rank=0)) is None:
                return check_result.fail(
                    "Query scale is not a scalar.",
                    query_scale,
                )
        else:
            assert query_scaling == "Div", "Unexpected query scaling operation"
            if (query_scale_value := _ir_utils.get_singleton_value(query_scale, rank=0)) is None:
                return check_result.fail(
                    "Query scale is not a scalar.",
                    query_scale,
                )
            query_scale_value = 1.0 / query_scale_value

        if key_scaling == "None":
            key_scale_value = 1.0
        elif key_scaling == "Mul":
            if (key_scale_value := _ir_utils.get_singleton_value(key_scale, rank=0)) is None:
                return check_result.fail(
                    "Key scale is not a scalar.",
                    key_scale,
                )
        else:
            assert key_scaling == "Div", "Unexpected key scaling operation"
            if (key_scale_value := _ir_utils.get_singleton_value(key_scale, rank=0)) is None:
                return check_result.fail(
                    "Key scale is not a scalar.",
                    key_scale,
                )
            key_scale_value = 1.0 / key_scale_value

        if qk_scale == "None":
            qk_scale_value = 1.0
        elif qk_scale == "Mul":
            if (qk_scale_value := _ir_utils.get_singleton_value(qk_scale, rank=0)) is None:
                return check_result.fail(
                    "QK scale is not a scalar.",
                    qk_scale,
                )
        else:
            assert qk_scale == "Div", "Unexpected QK scaling operation"
            if (qk_scale_value := _ir_utils.get_singleton_value(qk_scale, rank=0)) is None:
                return check_result.fail(
                    "QK scale is not a scalar.",
                    qk_scale,
                )
            qk_scale_value = 1.0 / qk_scale_value

        self._scale = query_scale_value * key_scale_value * qk_scale_value            

        # If the scaling factor is the default one, we can skip passing it to SDPA.

        if query is None or query.shape is None or len(query.shape) < 2:
            return
        hidden_size = query.shape[-1]
        if not isinstance(hidden_size, int):
            return

        default_scaling_factor = math.sqrt(hidden_size)

        if self._scale == default_scaling_factor:
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
