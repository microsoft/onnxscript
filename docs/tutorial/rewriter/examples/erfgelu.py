# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Onnx Pattern Rewriting.

This script shows how to define a rewriting rule based on patterns.

First a dummy model with a GELU activation
===================
"""

import math

import onnx

import onnxscript
from onnxscript import FLOAT, ir, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(X: FLOAT[64, 128], Y: FLOAT[64, 128]) -> FLOAT[64, 128]:
    input_add = opset18.Add(X, Y)
    sqrt2 = opset18.Constant(value_float=math.sqrt(2))
    erf = opset18.Erf(input_add / sqrt2)
    add_const = opset18.Constant(value_float=1.0)
    plus_one = erf + add_const
    mul1 = input_add * plus_one
    mul_const = opset18.Constant(value_float=0.5)
    result = mul_const * mul1
    return result


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# Model demonstrating multiple patterns and variations of GELU activation
# =====================


@script()
def commute_model(X: FLOAT[64, 128], Y: FLOAT[64, 128]) -> FLOAT[64, 128]:
    # Create first GELU variant
    sqrt2_v1 = opset18.Constant(value_float=math.sqrt(2))
    erf_v1 = opset18.Erf(X / sqrt2_v1)
    add_const_v1 = opset18.Constant(value_float=1.0)
    plus_one_v1 = erf_v1 + add_const_v1
    mul1_v1 = X * plus_one_v1
    mul_const_v1 = opset18.Constant(value_float=0.5)
    gelu1 = mul_const_v1 * mul1_v1

    # Create second GELU variant
    sqrt2_v2 = opset18.Constant(value_float=math.sqrt(2))
    erf_v2 = opset18.Erf(Y / sqrt2_v2)
    add_const_v2 = opset18.Constant(value_float=1.0)
    plus_one_v2 = erf_v2 + add_const_v2
    mul1_v2 = Y * plus_one_v2
    mul_const_v2 = opset18.Constant(value_float=0.5)
    gelu2 = mul1_v2 * mul_const_v2

    # Add both GELU functions
    result = opset18.Add(gelu1, gelu2)
    return result


commute_model = commute_model.to_model_proto()
onnx.checker.check_model(commute_model)


####################################
# The target pattern
# =====================


def erf_gelu_pattern(op, x):
    return 0.5 * (x * (op.Erf(x / math.sqrt(2)) + 1.0))


def erf_gelu_pattern_2(op, x):
    return (x * (op.Erf(x / math.sqrt(2)) + 1.0)) * 0.5


####################################
# The replacement pattern
# =====================


def gelu(op, x: ir.Value):
    return op.Gelu(x, _domain="com.microsoft")


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    rule = pattern.RewriteRule(
        erf_gelu_pattern,  # Target Pattern
        gelu,  # Replacement
    )
    model_with_rewrite_applied = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[rule],
    )
    return model_with_rewrite_applied


def apply_rewrite_with_ruleset(model):
    # Create multiple rules
    rule1 = pattern.RewriteRule(
        erf_gelu_pattern,  # Target Pattern
        gelu,  # Replacement
    )
    rule2 = pattern.RewriteRule(
        erf_gelu_pattern_2,  # Target Pattern
        gelu,  # Replacement
    )
    # Create a Rewrite Rule Set with multiple rules.
    rewrite_rule_set = pattern.RewriteRuleSet([rule1, rule2])
    # Apply rewrites
    model_with_rewrite_applied = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
        # pattern_rewrite_rules=[rule1, rule2], # Alternative method of passing multiple rules
    )
    return model_with_rewrite_applied


def apply_rewrite_with_commute(model):
    rule = pattern.RewriteRule(
        erf_gelu_pattern,  # Target Pattern
        gelu,  # Replacement
    )
    # Create a Rewrite Rule Set with commute=True
    rewrite_rule_set = pattern.RewriteRuleSet([rule], commute=True)
    # Apply rewrites
    model_with_rewrite_applied = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite_applied


# Rewrite-Simple
model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(model_with_rewrite)

# Rewrite-Single-Patterns
# Incorrect number of rewrites
model_with_single_rewrite_ruleset = apply_rewrite(commute_model)
onnx.checker.check_model(model_with_single_rewrite_ruleset)

# Rewrite-Multiple-Patterns-RuleSet
model_with_rewrite_ruleset = apply_rewrite_with_ruleset(commute_model)
onnx.checker.check_model(model_with_rewrite_ruleset)

# Rewrite-Multiple-Patterns-Commute
model_with_rewrite_commute = apply_rewrite_with_commute(commute_model)
onnx.checker.check_model(model_with_rewrite_commute)
