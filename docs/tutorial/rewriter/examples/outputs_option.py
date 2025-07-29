# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ONNX Pattern Rewriting with output specification

This script shows how to define a rewriting rule that specifies
the number and names of outputs from operations.
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(A: FLOAT[4, 4]) -> FLOAT[2, 4]:
    # Split operation that produces 2 outputs
    result1, _result2 = opset18.Split(A, num_outputs=2, axis=0)
    # We only return the first output for simplicity
    return result1


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern with multiple outputs
# =====================


def split_pattern(op, input):
    # Pattern to match Split operations with 2 outputs
    # num_outputs=2 corresponds to the attribute of the ONNX Split op
    # _outputs=2 is an option controlling the pattern constructor
    return op.Split(input, num_outputs=2, axis=0, _outputs=2)


####################################
# The replacement pattern with named outputs
# =====================


def custom_split_replacement(op, input, **_):
    # Replace with a custom split operation using named outputs
    # _outputs=["first_half", "second_half"] assigns names to the outputs
    # IMPORTANT: The number of outputs must match the pattern (2 outputs)
    return op.CustomSplit(
        input, _domain="custom.domain", _outputs=["first_half", "second_half"]
    )


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    # Create rewrite rules
    split_rule = pattern.RewriteRule(
        split_pattern,  # target pattern - matches Split with 2 outputs
        custom_split_replacement,  # replacement pattern - uses named outputs
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([split_rule])
    # Apply rewrite
    model_with_rewrite = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite


_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)
