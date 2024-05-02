"""Onnx Pattern Rewriting with match condition parameter.

This script shows how to define a rewriting rule based on patterns while
utilizing the match condition parameter.

First we write a dummy model with a several Reshape nodes and a Matmul node
===================
"""

import logging

import numpy as np
import onnx

import onnxscript
from onnxscript import FLOAT, ir, opset18, script
from onnxscript.rewriter import _ir_utils, pattern

logger = logging.getLogger(__name__)


@script()
def original_model(A: FLOAT[1, 4, 512, 512], B: FLOAT[1, 4, 512, 64]) -> FLOAT[1, 4, 512, 64]:
    # NOTE: Modified from `value_ints` to `value`
    shape_a = opset18.Constant(value=[4, 512, 512])
    reshape_a = opset18.Reshape(A, shape_a)
    shape_b = opset18.Constant(value=[4, 512, 64])
    reshape_b = opset18.Reshape(B, shape_b)
    matmul = opset18.MatMul(reshape_a, reshape_b)
    shape_c = opset18.Constant(value=[1, 4, 512, 64])
    result = opset18.Reshape(matmul, shape_c)
    return result


_model = original_model.to_model_proto()
onnx.checker.check_model(_model)


####################################
# The target pattern
# =====================

_op = pattern.onnxop


def two_reshapes_matmul_reshape_pattern(input_a, input_b, shape_a, shape_b, shape_c):
    reshape_a = _op.Reshape(input_a, shape_a)
    reshape_b = _op.Reshape(input_b, shape_b)
    matmul = _op.MatMul(reshape_a, reshape_b)
    return _op.Reshape(matmul, shape_c)


####################################
# The replacement pattern
# =====================


def matmul_pattern(op, input_a: ir.Value, input_b: ir.Value, **_):
    return op.MatMul(input_a, input_b)


####################################
# Write condition to check if we need to replace the pattern
# =====================


def check_if_need_reshape(input_a, input_b, shape_c, **_) -> bool:
    """If matmul broadcasting is enough, then we don't need the reshapes.

    To validate this, we need to check the following:
    1. Input shapes check: input_a and input_b should be broadcastable
    2. Output shape check: shape_c should be the same as the output shape from the matmul(input_a, input_b)

    If the above are true, then we don't need the reshapes.
    """
    input_a_shape = input_a.shape
    input_b_shape = input_b.shape
    # TODO: Get a helper func to get const_value
    shape_c_value = _ir_utils.propagate_const_value(shape_c)
    shape_c = shape_c_value.const_value.numpy()  # type: ignore[union-attr]
    if shape_c is None:
        return False
    if not isinstance(shape_c, np.ndarray):
        logger.info("Unexpected shape_c value. Expected np.ndarray, got %s", type(shape_c))
        return False
    if len(shape_c.shape) != 1:
        logger.info(
            "Unexpected final shape. The shape of 'shape' value is %s",
            shape_c.shape,
        )
        return False
    shape_c = shape_c.tolist()

    # NOTE: When there is a subset match with a pattern. The MatchResult won't have the shape
    # information. So, we need to check if the shape is None and return False.
    if input_a_shape is None or input_b_shape is None or shape_c is None:
        logger.info("Shape information is not available for the inputs and outputs.")
        return False
    input_a_shape = list(input_a_shape)
    input_b_shape = list(input_b_shape)

    dim_a = len(input_a_shape)
    dim_b = len(input_b_shape)

    # 1. Check if input shapes are broadcastable
    # 1.a. If the first input is 1-D, check whether
    # the dim matches the last second dim of the second input.
    mimic_matmul_broadcast_behavior = False
    if dim_a < 2:
        if input_a_shape[-1] != input_b_shape[-2]:
            logger.info("Original shape is not MatMul compatible.")
            return False
        else:
            input_a_shape = [1, *input_a_shape]
            dim_a = len(input_a_shape)
            mimic_matmul_broadcast_behavior = True
    # 1.b. If the second input is 1-D, check whether
    # the dim matches the last dim of the first input.
    if dim_b < 2:
        if input_b_shape[-1] != input_a_shape[-1]:
            logger.info("Original shape is not MatMul compatible.")
            return False
        else:
            input_b_shape = [*input_b_shape, 1]
            dim_b = len(input_b_shape)
            mimic_matmul_broadcast_behavior = True
    # 1.c. If both inputs are at least 2-D, check whether
    # the last dimension of the first input matches the second
    # last dimension of the second input, and shape[:-2] are
    # broadcastable.
    input_a_shape_except_second_last_dim = input_a_shape[:-2] + [input_a_shape[-1]]
    input_b_shape_except_last_dim = input_b_shape[:-1]
    broadcast_matmul_output_shape = [input_a_shape[-2], input_b_shape[-1]]
    for idx, (dim_from_a, dim_from_b) in enumerate(
        zip(
            reversed(input_a_shape_except_second_last_dim),
            reversed(input_b_shape_except_last_dim),
        )
    ):
        if dim_from_a not in {1, dim_from_b}:
            logger.info("Original shape is not broadcastable.")
            return False
        elif idx > 0:
            broadcast_matmul_output_shape = [
                max(dim_from_a, dim_from_b),
                *broadcast_matmul_output_shape,
            ]

    # 2. Check if output shape is the same as the output shape from the matmul(input_a, input_b)
    # Prepend the broadcast_matmul_output_shape with the longer shape of input
    if dim_a > dim_b:
        longer_shape = input_a_shape
        shorter_shape = input_b_shape
    else:
        longer_shape = input_b_shape
        shorter_shape = input_a_shape
    broadcast_matmul_output_shape = (
        longer_shape[: -len(shorter_shape)] + broadcast_matmul_output_shape
    )
    if mimic_matmul_broadcast_behavior and dim_b == 2:
        broadcast_matmul_output_shape = broadcast_matmul_output_shape[:-1]
    if mimic_matmul_broadcast_behavior and dim_a == 2:
        broadcast_matmul_output_shape.pop(-2)
    if shape_c != broadcast_matmul_output_shape:
        logger.info(
            "Final output shape is not the same. Expected %s vs actual %s",
            shape_c,
            broadcast_matmul_output_shape,
        )
        return False

    return True


####################################
# Create Rewrite Rule and Apply to Model
# =====================


def apply_rewrite(model):
    # Create rewrite rules
    two_reshapes_matmul_reshape_rule = pattern.RewriteRule(
        two_reshapes_matmul_reshape_pattern,  # target pattern
        matmul_pattern,  # replacement pattern
        check_if_need_reshape,  # match_condition function
    )
    # Create a Rewrite Rule Set
    rewrite_rule_set = pattern.RewriteRuleSet([two_reshapes_matmul_reshape_rule])
    # Apply rewrite while passing match_condition
    model_with_rewrite = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite


_model_with_rewrite = apply_rewrite(_model)
onnx.checker.check_model(_model_with_rewrite)
