# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ONNX Pattern Rewriting with domain specification

This script shows how to define a rewriting rule that targets operations
from specific domains and replaces them with operations in other domains.
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern


@script()
def original_model(A: FLOAT[2, 2]) -> FLOAT[2, 2]:
    # This would represent a custom operation in a specific domain
    # For demonstration, we'll use a standard Relu but imagine it's in a custom domain
    result = opset18.Relu(A)
    return result


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern
# =====================


def custom_relu_pattern(op, input):
    # Pattern to match Relu operations from a specific domain
    # _domain="custom.domain" specifies we only want to match operations from this domain
    return op.Relu(input, _domain="custom.domain")


####################################
# The replacement pattern
# =====================


def standard_relu_replacement(op, input, **_):
    # Replace with standard ONNX Relu (default domain)
    return op.Relu(input)


####################################
# Alternative: Replace with operation in different domain
# =====================


def microsoft_relu_replacement(op, input, **_):
    # Replace with operation in Microsoft's domain
    return op.OptimizedRelu(input, _domain="com.microsoft")


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    # Create rewrite rules
    relu_rule = pattern.RewriteRule(
        custom_relu_pattern,  # target pattern - matches custom domain operations
        standard_relu_replacement,  # replacement pattern - uses standard domain
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([relu_rule])
    # Apply rewrite
    model_with_rewrite = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite


# Note: This example is demonstrative. In practice, you would modify the model
# to have operations in the custom domain before applying the rewrite.
_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)