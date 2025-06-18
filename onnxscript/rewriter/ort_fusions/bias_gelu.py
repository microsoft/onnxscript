# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern


class BiasGeluFusion(pattern.RewriteRuleClassBase):
    """Fuses a Bias-Gelu pattern into a single BiasGelu operator.

    Attributes:
        contrib_op (bool): If True, matches the Gelu operator from the 'com.microsoft' domain.
            If False, matches the standard ONNX Gelu operator.
    """

    def __init__(
        self,
        name: str,
        *,
        contrib_op: bool,
    ):
        super().__init__(name)
        self._contrib_op = contrib_op

    def pattern(self, op, input, bias):
        gelu_add = op.Add(input, bias)

        if self._contrib_op:
            return op.Gelu(gelu_add, _domain="com.microsoft", _outputs=["gelu"])
        else:
            return op.Gelu(gelu_add, _outputs=["gelu"])

    def check(self, op, gelu, input, bias, **_) -> pattern.MatchResult:
        check_result = pattern.MatchResult()
        approximate = gelu.producer().attributes.get_string("approximate")
        if approximate is not None and approximate == "tanh":
            return check_result.fail(
                "Gelu operator with 'approximate' set to 'tanh' is not supported."
            )

        if not _ir_utils.has_rank(bias, 1):
            return check_result.fail("bias is not of shape 1D tensor", bias)

        return check_result

    def rewrite(self, op, input, bias, **_):
        return op.BiasGelu(input, bias, _domain="com.microsoft")


bias_gelu_rules = pattern.RewriteRuleSet(
    [
        *BiasGeluFusion.rule("gelu_onnx_op", contrib_op=False).commute(),
        *BiasGeluFusion.rule("gelu_contrib_op", contrib_op=True).commute(),
    ]
)


fuse_bias_gelu = _fusion_utils.apply_fusion_rules(bias_gelu_rules)
