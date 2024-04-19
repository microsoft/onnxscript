"""Onnx Pattern Rewriting.

This script shows how to define a rewriting rule based on patterns.

First a dummy model with a GELU activation
===================
"""


import math
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

import onnxscript
from onnxscript.rewriter import pattern


def original_model():
    inputs = [
        oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape=[]),
        oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape=[]),
    ]
    nodes = [
        oh.make_node("Add", ["x", "y"], ["_onx_add0"]),
        oh.make_node("Constant", inputs=[], outputs=['_onx_const0'], value_float=math.sqrt(2)),
        oh.make_node("Div", ["_onx_add0", "_onx_const0"], ["_onx_div0"]),
        oh.make_node("Erf", ["_onx_div0"], ["_onx_erf0"]),
        oh.make_node("Constant", inputs=[], outputs=['_onx_const1'], value_float=1.0),
        oh.make_node("Add", ["_onx_erf0", "_onx_const1"], ["_onx_add1"]),
        oh.make_node("Mul", ["_onx_add0", "_onx_add1"], ["_onx_mul0"]),
        oh.make_node("Constant", inputs=[], outputs=['_onx_const2'], value_float=0.5),
        oh.make_node("Mul", ["_onx_const2", "_onx_mul0"], ["_onx_mul1"]),

    ]
    outputs = [
        oh.make_tensor_value_info("_onx_mul1", onnx.TensorProto.FLOAT, []),
    ]
    model = oh.make_model(
        oh.make_graph(
            nodes,
            "experiment",
            inputs,
            outputs,
        ),
        opset_imports=[
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 18),
        ],
    )
    return model


model = original_model()
onnx.checker.check_model(model)


####################################
# The target pattern
# =====================

op = pattern.onnxop
msft_op = pattern.msft_op


def erf_gelu_pattern(x):
    return 0.5 * (x * (op.Erf(x / math.sqrt(2)) + 1.0))


####################################
# The replacement pattern
# =====================


def gelu(x):
    return msft_op.Gelu(x)


####################################
# Create Rewrite Rule and Apply to Model
# =====================

rule = pattern.RewriteRule(
    erf_gelu_pattern,       # Target Pattern
    gelu,                   # Replacement Pattern
)
model_with_rewrite = onnxscript.rewriter.rewrite(
    model,
    pattern_rewrite_rules=[rule],
)
        
onnx.checker.check_model(model_with_rewrite)
