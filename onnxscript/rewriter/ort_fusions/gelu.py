# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript.rewriter import _fusion_utils, pattern

_SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
_SQRT_TWO = math.sqrt(2.0)


class GeluTanhFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x):
        # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
        t1 = op.Pow(x, 3)
        t2 = op.Mul(0.044715, t1)
        t3 = op.Add(x, t2)

        t4 = op.Mul(_SQRT_TWO_OVER_PI, t3)
        t5 = op.Tanh(t4)
        t6 = op.Add(t5, 1)
        t7 = op.Mul(0.5, t6)
        result = op.Mul(x, t7)
        return result

    def rewrite(self, op, x):
        return op.FastGelu(x, _domain="com.microsoft")


class GeluErfFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x):
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        t1 = op.Div(x, _SQRT_TWO)
        t2 = op.Erf(t1)
        t3 = op.Add(t2, 1.0)
        t4 = op.Mul(x, t3)
        result = op.Mul(t4, 0.5)
        return result

    def rewrite(self, op, x):
        return op.Gelu(x, _domain="com.microsoft")


_tanh_rule = GeluTanhFusion.rule()
_erf_rule = GeluErfFusion.rule()

gelu_rules = pattern.RewriteRuleSet([_tanh_rule, _erf_rule])

fuse_gelu = _fusion_utils.apply_fusion_rules(gelu_rules)
