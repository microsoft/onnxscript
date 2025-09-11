# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ONNX Pattern Rewriting with variable number of inputs

This script shows how to define a rewriting rule based on patterns that
can match nodes with additional inputs beyond those specified in the pattern.
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(A: FLOAT[2, 2], B: FLOAT[2, 2], C: FLOAT[2, 2]) -> FLOAT[2, 2]:
    # Conv with bias - has 3 inputs: input, weight, bias
    result = opset18.Conv(A, B, C)
    return result


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern
# =====================


def conv_pattern(op, input, weight):
    # Pattern to match Conv operations, allowing additional inputs like bias
    # _allow_other_inputs=True allows the pattern to match Conv with bias (3 inputs)
    # even though we only specify 2 inputs in the pattern
    return op.Conv(input, weight, _allow_other_inputs=True)


####################################
# The replacement pattern
# =====================


def conv_replacement(op, input, weight, **_):
    # Replace with a custom operation in a different domain
    return op.OptimizedConv(input, weight, _domain="custom.domain")


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    # Create rewrite rules
    conv_rule = pattern.RewriteRule(
        conv_pattern,  # target pattern
        conv_replacement,  # replacement pattern
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([conv_rule])
    # Apply rewrite
    model_with_rewrite = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite


_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)
