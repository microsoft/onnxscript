# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Onnx Pattern Rewriting with _allow_other_attributes option.

This script shows how to define a rewriting rule based on patterns while
utilizing the _allow_other_attributes option.

First we write a dummy model with an Add node with an additional attribute
===================
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(A: FLOAT[2, 2], B: FLOAT[2, 2]) -> FLOAT[2, 2]:
    add = opset18.Add(A, B, alpha=0.5)
    return add


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern
# =====================


def add_pattern(op, input_a, input_b):
    return op.Add(input_a, input_b, _allow_other_attributes=True)


####################################
# The replacement pattern
# =====================


def add_replacement(op, input_a, input_b, **_):
    return op.Add(input_a, input_b)


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    # Create rewrite rules
    add_rule = pattern.RewriteRule(
        add_pattern,  # target pattern
        add_replacement,  # replacement pattern
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([add_rule])
    # Apply rewrite while passing match_condition
    model_with_rewrite = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite


_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)
