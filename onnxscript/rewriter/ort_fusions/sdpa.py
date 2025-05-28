# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript import ir
from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern


class SDPA(pattern.RewriteRuleClassBase):
    _scale: float | None

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
    ):
        # Some implementations scale the query and key before computing the dot product
        query = pattern.OrValue(
            [
                op.Mul(query, query_scale),
                op.Div(query, query_scale),
                query,
            ],
            tag_var="query_scaling",
            tag_values=["Mul", "Div", "None"],
        )
        key_transposed = pattern.OrValue(
            [
                op.Mul(key_transposed, key_scale),
                op.Div(key_transposed, key_scale),
                key_transposed,
            ],
            tag_var="key_scaling",
            tag_values=["Mul", "Div", "None"],
        )

        attn_score = op.MatMul(query, key_transposed)

        # Some implementations scale the dot product.
        attn_score = pattern.OrValue(
            [
                op.Mul(attn_score, qk_scale),
                op.Div(attn_score, qk_scale),
                attn_score,
            ],
            tag_var="qk_scaling",
            tag_values=["Mul", "Div", "None"],
        )

        # Some implementations add a mask to the dot product.
        masked_attn_score = op.Add(attn_score, mask)
        attn_score = pattern.OrValue(
            [masked_attn_score, attn_score], tag_var="has_mask", tag_values=[True, False]
        )

        attn_weight = op.Softmax(attn_score, axis=-1)
        attn_output = op.MatMul(attn_weight, value)
        return attn_output

    def check(
        self,
        context,
        query: ir.Value | None,
        query_scaling: str,
        query_scale: ir.Value | None,
        key_scaling: str,
        key_scale: ir.Value | None,
        qk_scaling: str,
        qk_scale: ir.Value | None,
        **_,
    ):
        check_result = pattern.MatchResult()

        if query_scaling == "None":
            query_scale_value = 1.0
        elif query_scaling == "Mul":
            if (query_scale_value := _ir_utils.get_singleton_value(query_scale)) is None:
                return check_result.fail(
                    "Query scale is not a scalar.",
                    query_scale,
                )
        else:
            assert query_scaling == "Div", "Unexpected query scaling operation"
            if (query_scale_value := _ir_utils.get_singleton_value(query_scale)) is None:
                return check_result.fail(
                    "Query scale is not a scalar.",
                    query_scale,
                )
            query_scale_value = 1.0 / query_scale_value

        if key_scaling == "None":
            key_scale_value = 1.0
        elif key_scaling == "Mul":
            if (key_scale_value := _ir_utils.get_singleton_value(key_scale)) is None:
                return check_result.fail(
                    "Key scale is not a scalar.",
                    key_scale,
                )
        else:
            assert key_scaling == "Div", "Unexpected key scaling operation"
            if (key_scale_value := _ir_utils.get_singleton_value(key_scale)) is None:
                return check_result.fail(
                    "Key scale is not a scalar.",
                    key_scale,
                )
            key_scale_value = 1.0 / key_scale_value

        if qk_scaling == "None":
            qk_scale_value = 1.0
        elif qk_scaling == "Mul":
            if (qk_scale_value := _ir_utils.get_singleton_value(qk_scale)) is None:
                return check_result.fail(
                    "QK scale is not a scalar.",
                    qk_scale,
                )
        else:
            assert qk_scaling == "Div", "Unexpected QK scaling operation"
            if (qk_scale_value := _ir_utils.get_singleton_value(qk_scale)) is None:
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

        default_scaling_factor = 1.0 / math.sqrt(hidden_size)

        if math.isclose(self._scale, default_scaling_factor, rel_tol=1e-5, abs_tol=1e-8):
            # Pass no scaling factor to SDPA, SDPA will use the default scaling factor
            self._scale = None

        # TODO: check ranks/shapes

        return check_result

    def rewrite(
        self,
        op,
        query: ir.Value | None,
        key_transposed: ir.Value | None,
        value: ir.Value | None,
        mask: ir.Value | None,
        **_,
    ):
        sdpa_args = [query, key_transposed, value]
        if mask is not None:
            sdpa_args.append(mask)
        return op.SDPA(*sdpa_args, scale=self._scale, _domain="ai.onnxruntime.fusion")


# Dynamically create the rules
sdpa_rules = pattern.RewriteRuleSet([SDPA.rule()])


fuse_sdpa = _fusion_utils.apply_fusion_rules(sdpa_rules)
