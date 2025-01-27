# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math

from onnxscript.rewriter import pattern


# Pattern to match against
def erf_gelu_pattern(op, x):
    # erf_gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # half = pattern.Constant(0.5)
    # sqrt2 = pattern.Constant(1.4142)
    # x_div_sqrt2 = op.Div(x, sqrt2)
    # erf = op.Erf(x_div_sqrt2)
    # one = pattern.Constant(1.0)
    # one_plus_erf = op.Add(erf, one)
    # x_mul_one_plus_erf = op.Mul(x, one_plus_erf)
    # return op.Mul(half, x_mul_one_plus_erf)
    return 0.5 * (x * (op.Erf(x / math.sqrt(2)) + 1.0))


# Replacement
def gelu(op, x):
    return op.Gelu(x, _domain="com.microsoft")


rule = pattern.RewriteRule(erf_gelu_pattern, gelu)
