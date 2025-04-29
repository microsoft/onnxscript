# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, pattern


class BiasGeluFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, y):
        gelu_add = op.Add(x, y)
        return op.Gelu(gelu_add, _domain="com.microsoft")

    def rewrite(self, op, x, y):
        return op.BiasGelu(x, y, _domain="com.microsoft")


_rule = BiasGeluFusion.rule()

bias_gelu_rules = pattern.RewriteRuleSet([_rule])


fuse_bias_gelu = _fusion_utils.apply_fusion_rules(bias_gelu_rules)
