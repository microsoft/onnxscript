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
        cubed = op.Pow(x, 3)
        inner = op.Mul(0.044715, cubed)
        inner = op.Add(x, inner)

        inner = op.Mul(_sqrt_two_over_pi, inner)
        inner = op.Tanh(inner)
        inner = op.Add(inner, 1)
        inner = op.Mul(x, inner)
        result = op.Mul(0.5, inner)
        return result

    def rewrite(self, op, x):
        return op.Gelu(x, _domain="com.microsoft")


_rule = GeluTanhFusion.rule()

gelu_rules = pattern.RewriteRuleSet([_rule])


def fuse_gelu(model: ir.Model) -> None:
    gelu_rules.apply_to_model(model)
