# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Onnx Pattern Rewriting with attributes

This script shows how to define a rewriting rule based on patterns that
are dependent on the attributes of the nodes.
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(A: FLOAT[2, 2], B: FLOAT[2, 2]) -> FLOAT[2, 2]:
    add = opset18.Add(A, B)
    result = opset18.Dropout(add, training_mode=False)
    return result


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern
# =====================


def add_pattern(op, input):
    return op.Dropout(input, training_mode=False, _allow_other_attributes=True)


####################################
# The replacement pattern
# =====================


def add_replacement(op, input, **_):
    return op.Identity(input)


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
