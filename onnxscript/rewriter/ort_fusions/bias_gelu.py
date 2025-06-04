# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, pattern


class BiasGeluFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name,
        *,
        has_contrib_op: bool,
    ):
        super().__init__(name)
        self._has_contrib_op = has_contrib_op

    def pattern(self, op, x, y):
        gelu_add = op.Add(x, y)
        # see: gh-2362. Match against Gelu from onnx op or contrib ops.
        if self._has_contrib_op:
            return op.Gelu(gelu_add, _domain="com.microsoft")
        else:
            return op.Gelu(gelu_add)

    def rewrite(self, op, x, y):
        return op.BiasGelu(x, y, _domain="com.microsoft")


bias_gelu_rules = pattern.RewriteRuleSet(
    [
        BiasGeluFusion.rule("gelu_onnx_op", has_contrib_op=False),
        BiasGeluFusion.rule("gelu_contrib_op", has_contrib_op=True),
    ]
)


fuse_bias_gelu = _fusion_utils.apply_fusion_rules(bias_gelu_rules)
