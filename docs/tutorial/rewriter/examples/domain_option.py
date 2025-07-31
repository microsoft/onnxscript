# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ONNX Pattern Rewriting with domain specification

This script shows how to define a rewriting rule that targets operations
from specific domains and replaces them with operations in other domains.
"""

import onnx

import onnxscript
from onnxscript import script
from onnxscript.rewriter import pattern
from onnxscript.values import Opset

# Create an opset for the custom domain
opset = Opset("custom.domain", 1)


@script(opset)
def create_model_with_custom_domain(input: onnxscript.FLOAT[2, 2]) -> onnxscript.FLOAT[2, 2]:
    """Create a model with a Relu operation in a custom domain."""
    return opset.Relu(input)


_model = create_model_with_custom_domain.to_model_proto()
_model = onnx.shape_inference.infer_shapes(_model)
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


# The rewrite rule will now match the Relu operation in the custom domain
# and replace it with a standard ONNX Relu operation
_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)
