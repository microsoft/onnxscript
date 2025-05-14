# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OR-patterns.

This script shows how to define a rewriting rule based on OR-patterns.
"""

import onnx

import onnxscript
from onnxscript import FLOAT, opset18, script
from onnxscript.rewriter import pattern

####################################
# The target pattern
# =====================


def scaled_matmul(op, x, y, factor):
    xy = op.MatMul(x, y)
    choice1 = op.Mul(xy, factor)
    choice2 = op.Div(xy, factor)
    scaled_xy = pattern.OrValue(
        [choice1, choice2], tag_var="op_type", tag_values=["Mul", "Div"]
    )
    return op.Relu(scaled_xy)


####################################
# The replacement pattern
# =====================


def scaled_matmul_replacement(op, x, y, factor, op_type):
    if op_type == "Mul":
        return op.MatMulMulRelu(x, y, factor, _domain="some.domain")
    elif op_type == "Div":
        return op.MatMulDivRelu(x, y, factor, _domain="some.domain")
    else:
        raise ValueError(f"Unknown operation type: {op_type}")


####################################
# Rewrite Rule
# =====================
def apply_rewrite(model):
    rule = pattern.RewriteRule(
        scaled_matmul,  # target pattern
        scaled_matmul_replacement,  # replacement pattern
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([rule])
    return onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )


@script()
def original_model1(A: FLOAT[2, 2], B: FLOAT[2, 2]) -> FLOAT[2, 2]:
    t1 = opset18.MatMul(A, B)
    c = opset18.Constant(value_float=2.0)
    t2 = opset18.Mul(t1, c)
    t3 = opset18.Relu(t2)
    return t3


_model = original_model1.to_model_proto()
onnx.checker.check_model(_model)

_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)

assert [n.op_type for n in _model_with_rewrite.graph.node] == ["Constant", "MatMulMulRelu"]


@script()
def original_model2(A: FLOAT[2, 2], B: FLOAT[2, 2]) -> FLOAT[2, 2]:
    t1 = opset18.MatMul(A, B)
    c = opset18.Constant(value_float=2.0)
    t2 = opset18.Div(t1, c)
    t3 = opset18.Relu(t2)
    return t3


_model = original_model2.to_model_proto()
onnx.checker.check_model(_model)

_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)

assert [n.op_type for n in _model_with_rewrite.graph.node] == ["Constant", "MatMulDivRelu"]
