# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern


class SDPA(pattern.RewriteRuleClassBase):
    def __init__(self, name: str, *, use_mask: bool, pre_scale: bool, use_mul: bool):
        super().__init__(name=name, as_function=True)
        self._use_mask = use_mask
        self._pre_scale = pre_scale
        self._use_mul = use_mul
        self._scale: float | None = None

    def pattern(
        self, op, query, key_transposed, value, mask, query_scale, key_scale, qk_scale
    ):
        if self._pre_scale:
            # Some implementations scale the query and key before computing the dot product
            if self._use_mul:
                query = op.Mul(query, query_scale)
                key_transposed = op.Mul(key_transposed, key_scale)
            else:
                query = op.Div(query, query_scale)
                key_transposed = op.Div(key_transposed, key_scale)
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

    def check(self, op, query, key_transposed, value, mask, query_scale, key_scale, qk_scale):
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

        if self._pre_scale:
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

    def rewrite(self, op, query, key_transposed, value, mask, **_):
        sdpa_args = [query, key_transposed, value]
        if self._use_mask:
            sdpa_args.append(mask)
        return op.SDPA(*sdpa_args, scale=self._scale, _domain="ai.onnxruntime.fusion")


# Rules for SDPA without mask
unmasked_pre_div_sdpa_rule = SDPA.rule(
    "unmasked_pre_div_sdpa", use_mask=False, pre_scale=True, use_mul=False
)
unmasked_pre_mul_sdpa_rule = SDPA.rule(
    "unmasked_pre_mul_sdpa", use_mask=False, pre_scale=True, use_mul=True
)
unmasked_post_div_sdpa_rule = SDPA.rule(
    "unmasked_post_div_sdpa", use_mask=False, pre_scale=False, use_mul=False
)
unmasked_post_mul_sdpa_rule = SDPA.rule(
    "unmasked_post_mul_sdpa", use_mask=False, pre_scale=False, use_mul=True
)

# Rules for SDPA with mask
masked_pre_div_sdpa_rule = SDPA.rule(
    "masked_pre_div_sdpa", use_mask=True, pre_scale=True, use_mul=False
)
masked_pre_mul_sdpa_rule = SDPA.rule(
    "masked_pre_mul_sdpa", use_mask=True, pre_scale=True, use_mul=True
)
masked_post_div_sdpa_rule = SDPA.rule(
    "masked_post_div_sdpa", use_mask=True, pre_scale=False, use_mul=False
)
masked_post_mul_sdpa_rule = SDPA.rule(
    "masked_post_mul_sdpa", use_mask=True, pre_scale=False, use_mul=True
)

sdpa_rules = pattern.RewriteRuleSet(
    [
        unmasked_pre_mul_sdpa_rule,
        unmasked_post_div_sdpa_rule,
        unmasked_post_mul_sdpa_rule,
        unmasked_pre_div_sdpa_rule,
        masked_pre_mul_sdpa_rule,
        masked_post_div_sdpa_rule,
        masked_post_mul_sdpa_rule,
        masked_pre_div_sdpa_rule,
    ]
)


fuse_sdpa = _fusion_utils.apply_fusion_rules(sdpa_rules)
