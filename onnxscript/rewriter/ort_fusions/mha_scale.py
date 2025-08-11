# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

"""
Multi-Head Attention (MHA) pre-scaling fusion patterns.

This module contains rewrite rules for fusing scale operations that occur before
Multi-Head Attention operations. The fusion optimizes patterns where a query tensor
is scaled before being passed to MHA by incorporating the scaling directly into
the MHA operation.

Example pattern:
    query -> Mul(scale) -> MultiHeadAttention -> output

Gets rewritten to:
    query -> MultiHeadAttention(with integrated scaling) -> output
"""


class FuseMHAScale(pattern.RewriteRuleClassBase):
    def pattern(self, op, query, scale):
        scaled_query = op.Mul(query, scale)
        mha_output = op.MultiHeadAttention(
            scaled_query,
            _allow_other_inputs=True,
            _domain="com.microsoft",
            _outputs=["mha_output"],
        )
        return mha_output

    def check(self, context, scale, **_):
        scale_value = _ir_utils.get_singleton_value(scale)
        if scale_value is None or not isinstance(scale_value, (int, float)):
            return pattern.MatchResult().fail("Scale must be a constant numeric value.", scale)
        self._scale = scale_value
        return True

    def rewrite(self, op, query, mha_output, **_):
        # Integrate the scale into the MHA operation
        mha_node = mha_output.producer()
        assert mha_node is not None
        # Compute original scale factor for MHA:
        attributes = mha_node.attributes
        original_scale = attributes.get_float("scale", None)
        if original_scale is None:
            num_heads = attributes.get_int("num_heads", None)
            if num_heads is None:
                return None
            head_size = query.shape[-1] // num_heads
            original_scale = 1.0 / math.sqrt(head_size)
        self._scale *= original_scale
        inputs = list(mha_node.inputs)
        inputs[0] = query
        attributes = dict(attributes)
        attributes["scale"] = self._scale
        return op.MultiHeadAttention(
            *inputs, **attributes, _domain="com.microsoft", _outputs=1
        )


_mha_scale_rules = pattern.RewriteRuleSet([FuseMHAScale.rule()])

fuse_mha_scale = _fusion_utils.apply_fusion_rules(_mha_scale_rules)
