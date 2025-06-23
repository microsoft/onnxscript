# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
from typing import Union

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern
from onnxscript.rewriter._basics import MatchFailureError

Dim = Union[int, ir.SymbolicDim]


class SDPA(pattern.RewriteRuleClassBase):
    _scale: float | None

    def pattern(
        self,
        op,
        query,
        key,
        value,
        mask,
        query_scale,
        key_scale,
        qk_scale,
    ):
        # The last two axes of key must be transposed before computing the dot product with query.
        # Three patterns are observed in practice:

        # Pattern 1: Transpose 4D key directly: BHSd => BHdS
        key_transposed_1 = op.Transpose(key, perm=[0, 1, 3, 2])

        # Pattern 2: Transpose key after converting to 3D and then convert back to 4D: BHSd => 3D => BHdS
        key_3d = op.Reshape(key, pattern.ANY_VALUE)
        key_3d_transposed = op.Transpose(key_3d, perm=[0, 2, 1])
        key_transposed_2 = op.Reshape(key_3d_transposed, pattern.ANY_VALUE)

        # Pattern 3: This transpose is sometimes composed with an earlier transpose to convert
        # the key from BSHd format to BHSd format.
        key_transposed_3 = op.Transpose(key, perm=[0, 2, 3, 1])

        key_transposed = pattern.OrValue(
            [key_transposed_1, key_transposed_2, key_transposed_3],
            tag_var="key_format",
            tag_values=["BHSd", "BHSd", "BSHd"],
        )

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
        key: ir.Value | None,
        value: ir.Value | None,
        mask: ir.Value | None,
        key_format: str,
        **match_bindings,
    ):
        check_result = pattern.MatchResult()

        bindings: dict[str, Dim] = {}

        # Check that query/key/value have the expected shapes:
        # They all should have same batch-size (B) and number of heads (H). Conceptually, it is
        # different for Q and K/V, but the certain op implementations require them to be the same,
        # which is usually achieved via tiling/expanding K/V num-heads to match Q num-heads.
        # Query and Key should have same head-size (Dh) while value can have different head-size (Dv).
        # Key and Value should have same sequence length (Skv), while Query can have different sequence length (S).
        _fusion_utils.check_shape(bindings, query, ["B", "H", "S", "Dh"])
        if key_format == "BHSd":
            _fusion_utils.check_shape(bindings, key, ["B", "H", "Skv", "Dh"])
        else:
            assert key_format == "BSHd", f"Unexpected key format: {key_format}"
            _fusion_utils.check_shape(bindings, key, ["B", "Skv", "H", "Dh"])
        _fusion_utils.check_shape(bindings, value, ["B", "H", "Skv", "Dv"])

        def get_scale_value(tag_name: str, scale_name: str) -> float:
            scaling_type = match_bindings.get(tag_name, "None")
            if scaling_type == "None":
                return 1.0
            else:
                scale = match_bindings.get(scale_name)
                value = _ir_utils.get_singleton_value(scale)
                if value is None:
                    raise MatchFailureError(f"{scale_name} is not a scalar.", scale)
                if scaling_type == "Mul":
                    return value
                else:
                    assert scaling_type == "Div", f"Unexpected {scale_name} scaling operation"
                    return 1.0 / value

        query_scale_value = get_scale_value("query_scaling", "query_scale")
        key_scale_value = get_scale_value("key_scaling", "key_scale")
        qk_scale_value = get_scale_value("qk_scaling", "qk_scale")

        self._scale = query_scale_value * key_scale_value * qk_scale_value

        # If the scaling factor is the default one, we can skip passing it to SDPA.

        head_size = bindings["Dh"]
        if not isinstance(head_size, int):
            return check_result

        default_scaling_factor = 1.0 / math.sqrt(head_size)

        if math.isclose(self._scale, default_scaling_factor, rel_tol=1e-5, abs_tol=1e-8):
            # Pass no scaling factor to SDPA, SDPA will use the default scaling factor
            self._scale = None

        return check_result

    def rewrite(
        self,
        op,
        query: ir.Value | None,
        key: ir.Value | None,
        value: ir.Value | None,
        mask: ir.Value | None,
        key_format: str,
        **_,
    ):
        sdpa_args = [query, key, value]
        if mask is not None:
            sdpa_args.append(mask)
        # If the scale is None, SDPA will use the default scaling factor, which is 1/sqrt(head_size).
        return op.SDPA(
            *sdpa_args,
            scale=self._scale,
            key_format=key_format,
            _domain="ai.onnxruntime._fusion",
        )


# Dynamically create the rules
sdpa_rules = pattern.RewriteRuleSet(
    [
        SDPA.rule(),
    ]
)

fuse_sdpa = _fusion_utils.apply_fusion_rules(sdpa_rules)
