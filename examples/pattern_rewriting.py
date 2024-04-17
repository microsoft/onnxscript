"""Onnx Pattern Rewriting.

This script shows how to define a rewriting rule based on patterns.
The objective is to replace some nodes in an onnx model into another
sequence of nodes but more efficient.

First a dummy model
===================
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

import onnxscript
from onnxscript import ir
from onnxscript.rewriter import generic_pattern


def get_rotary_model(bad_model=False):
    inputs = [
        oh.make_tensor_value_info("x", onnx.TensorProto.INT64, shape=[]),
        oh.make_tensor_value_info("pos_ids", onnx.TensorProto.FLOAT, shape=[]),
        oh.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=[]),
    ]
    nodes = [
        oh.make_node("Unsqueeze", ["x", "axis"], ["_onx_unsqueeze0"]),
        oh.make_node("Cast", ["_onx_unsqueeze0"], ["_onx_cast0"], to=1),
        oh.make_node("MatMul", ["pos_ids", "_onx_cast0"], ["_onx_matmul0"]),
        oh.make_node("Transpose", ["_onx_matmul0"], ["_onx_transpose0"]),
        oh.make_node(
            "ConcatTrainingBad" if bad_model else "ConcatTraining",
            ["_onx_transpose0", "_onx_transpose0"],
            ["_onx_concattraining0", "_onx_concattraining1"],
            domain="com.microsoft",
        ),
        oh.make_node("Sin", ["_onx_concattraining0"], ["_onx_sin0"]),
        oh.make_node("Cast", ["_onx_sin0"], ["_onx_cast02"], to=1),
        oh.make_node("Cos", ["_onx_concattraining0"], ["_onx_cos0"]),
        oh.make_node("Cast", ["_onx_cos0"], ["_onx_cast03"], to=1),
    ]
    outputs = [
        oh.make_tensor_value_info("_onx_cast02", onnx.TensorProto.UNDEFINED, []),
        oh.make_tensor_value_info("_onx_cast03", onnx.TensorProto.UNDEFINED, []),
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


model = get_rotary_model()
ir_model = ir.serde.deserialize_model(model)


####################################
# The rewriting pattern
# =====================

op = onnxscript.opset18
msft_op = onnxscript.values.Opset("com.microsoft", 1)


def rotary_match_pattern(x, pos_ids, axis):
    """The pattern to match."""
    unsqueeze = op.Unsqueeze(x, axis)
    cast = op.Cast(unsqueeze, to=onnx.TensorProto.FLOAT)

    matmul = op.MatMul(pos_ids, cast)
    transpose = op.Transpose(matmul)
    output, length = msft_op.ConcatTraining(transpose, transpose)

    sin = op.Sin(output)
    cast1 = op.Cast(sin, to=onnx.TensorProto.FLOAT)
    cos = op.Cos(output)
    cast2 = op.Cast(cos, to=onnx.TensorProto.FLOAT)
    return cast1, cast2


def validate_rotary_mapping(g, match_result) -> bool:
    """The validation post matching.

    Returns True to validate the replacement,
    False not to apply it.

    :param g: model
    :param match_result: matched nodes
    """
    del g
    del match_result
    return True


def rotary_apply_pattern(x, pos_ids, axis):
    """The replacement pattern."""
    cos_cache = op.Constant(value=onh.from_array(np.random.rand(256, 256).astype(np.float16)))
    sin_cache = op.Constant(value=onh.from_array(np.random.rand(256, 256).astype(np.float16)))
    part1, part2 = msft_op.RotaryEmbedding(x, pos_ids, cos_cache, sin_cache)
    return part1, part2


###########################
# The rule
# ========
#
# The rule is easy to create.


rule = generic_pattern.make_pattern_rule(
    rotary_match_pattern,
    rotary_apply_pattern,
    validate_rotary_mapping,
)

################################
# ``validate_rotary_mapping`` always return True.
# This argument can be ignored in that case.

rule = generic_pattern.make_pattern_rule(rotary_match_pattern, rotary_apply_pattern)

##########################
# Let's apply it.
rule.apply_to_model(ir_model)


########################
# And finally, we can generate the model.

rewritten_model = ir.serde.serialize_model(ir_model)

########################
# Let's see what it looks like.

for node in rewritten_model.graph.node:
    print(f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}")

#############################
# What if it fails?
# =================


model = get_rotary_model(True)
ir_model = ir.serde.deserialize_model(model)

rule.apply_to_model(ir_model)
rewritten_model = ir.serde.serialize_model(ir_model)

print([n.op_type for n in rewritten_model.graph.node])

################################
# The match did not happen.
# Let's increase the verbosity.

rule = generic_pattern.make_pattern_rule(
    rotary_match_pattern, rotary_apply_pattern, verbose=10
)

rule.apply_to_model(ir_model)

######################################
# The logs shows every time the algorithm rejected a pattern.
# We can see the following:
#
# ::
#
#     [OnnxGenericPattern.match] NONE - line: 673:onnxscript.rewriter.generic_pattern, op_type=Cast
#         --hint--: BACKWARD: different node types
#           --pattern
#           ConcatTraining(transpose, transpose) -> (output, length)
#           -- model
#           ConcatTrainingBad(_onx_transpose0, _onx_transpose0) -> (_onx_concattraining0, _onx_concattraining1)
#         iteration=1
#         --marked-- #2
#           Cast(_onx_cos0) ~ Cast(cos) [140186194226496-140186194222320]
#           Cos(_onx_concattraining0) ~ Cos(output) [140186194230816-140186194223472]
#         len(stacked)=0:[]
#
# Line 673 in file `generic_pattern.py`, the match was rejected.
# It says while comparing two nodes in the backward direction,
# node types do not match.
# It also says that two nodes were actually matched.
