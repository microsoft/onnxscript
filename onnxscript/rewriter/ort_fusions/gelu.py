# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math

from onnxscript import ir
from onnxscript.rewriter import pattern

_sqrt_two_over_pi = math.sqrt(2.0 / math.pi)


class GeluTanhFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x):
        # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
        t1 = op.Pow(x, 3)
        t2 = op.Mul(0.044715, t1)
        t3 = op.Add(x, t2)

        t4 = op.Mul(_sqrt_two_over_pi, t3)
        t5 = op.Tanh(t4)
        t6 = op.Add(t5, 1)
        t7 = op.Mul(x, t6)
        result = op.Mul(0.5, t7)
        return result

    def rewrite(self, op, x):
        return op.Gelu(x, _domain="com.microsoft")


_rule = GeluTanhFusion.rule()

gelu_rules = pattern.RewriteRuleSet([_rule])


def fuse_gelu(model: ir.Model) -> None:
    gelu_rules.apply_to_model(model)
