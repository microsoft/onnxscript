# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

# TODO: Support 1-D constant tensors
# https://github.com/microsoft/onnx-rewriter/issues/186


# Pattern to match against
def mul_by_1(op, x):
    return x * 1


def add_0(op, x):
    return x + 0


def sub_0(op, x):
    return x - 0


def div_by_1(op, x):
    return x / 1


def dropout_zero(op, x):
    return op.Dropout(x, ratio=0.0)


def dropout_inference(op, x):
    return op.Dropout(x, training_mode=False)


# Replacement
def identity(op, x, **_):
    return op.Identity(x)


mul_by_1_rule = RewriteRule(mul_by_1, identity)
add_0_rule = RewriteRule(add_0, identity)
sub_0_rule = RewriteRule(sub_0, identity)
div_by_1_rule = RewriteRule(div_by_1, identity)
dropout_zero_rule = RewriteRule(dropout_zero, identity)
dropout_inference_rule = RewriteRule(dropout_inference, identity)
# TODO: Include Mul by 0, 0 by Mul, 0 by Div? Those would be 0s, but not no-ops

rules = RewriteRuleSet(
    [
        *mul_by_1_rule.commute(),
        *add_0_rule.commute(),
        sub_0_rule,
        div_by_1_rule,
        dropout_zero_rule,
        dropout_inference_rule,
    ]
)
