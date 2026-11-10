# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule


def check_if_simulated_instance_norm_is_used(
    context,
    input_x,
    adjusted_input_shape,
    original_input_shape,
    weight_for_norm,
    bias_for_norm,
    weight_full,
    bias_full,
    **_,
) -> bool:
    """Recognize the static NCHW InstanceNormalization decomposition of GroupNorm."""
    if any(value.shape is None for value in (input_x, weight_full, bias_full)):
        return False
    weight_for_norm_const_value = weight_for_norm.const_value
    if weight_for_norm_const_value is None:
        return False
    weight_for_norm = weight_for_norm_const_value.numpy()

    bias_for_norm_const_value = bias_for_norm.const_value
    if bias_for_norm_const_value is None:
        return False
    bias_for_norm = bias_for_norm_const_value.numpy()

    if weight_for_norm.ndim != 1 or bias_for_norm.shape != weight_for_norm.shape:
        return False
    if not np.all(weight_for_norm == 1):
        return False
    if not np.all(bias_for_norm == 0):
        return False

    input_rank_minus_one = len(input_x.shape) - 1
    weight_full_rank = len(weight_full.shape)
    bias_full_rank = len(bias_full.shape)
    if weight_full_rank != input_rank_minus_one or bias_full_rank != input_rank_minus_one:
        return False

    input_rank = len(input_x.shape)
    if input_rank != 4:
        return False

    weight_full_shape = weight_full.shape
    if not all(dim == 1 for dim in weight_full_shape[1:]):
        return False
    bias_full_shape = bias_full.shape
    if not all(dim == 1 for dim in bias_full_shape[1:]):
        return False

    adjusted_input_shape_const_value = adjusted_input_shape.const_value

    g = weight_for_norm.shape[0]
    if (
        g <= 0
        or adjusted_input_shape_const_value is None
        or adjusted_input_shape_const_value.numpy().tolist() != [0, g, -1]
    ):
        return False

    # NOTE: Restrict the rule to only support constant shape
    original_input_shape_const_value = original_input_shape.const_value
    if (
        original_input_shape_const_value is None
        or original_input_shape_const_value.numpy().tolist() != input_x.shape
    ):
        return False

    return True


def instance_simulates_group_normalization_pattern(
    op,
    input_x,
    adjusted_input_shape,
    original_input_shape,
    weight_for_norm,
    bias_for_norm,
    weight_full,
    bias_full,
    epsilon,
):
    adjusted_input = op.Reshape(input_x, adjusted_input_shape)
    inst_norm = op.InstanceNormalization(
        adjusted_input, weight_for_norm, bias_for_norm, epsilon=epsilon
    )
    adjusted_inst_norm = op.Reshape(inst_norm, original_input_shape)
    mul = op.Mul(adjusted_inst_norm, weight_full)
    return op.Add(mul, bias_full)


def check_onnx_group_normalization(context, **kwargs) -> bool:
    if context.graph_or_function.opset_imports.get("", 0) < 21:
        return False
    input_x = kwargs["input_x"]
    weight_full = kwargs["weight_full"]
    bias_full = kwargs["bias_full"]
    # GroupNormalization's float32 stash would lose precision for DOUBLE inputs.
    if input_x.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        return False
    if not check_if_simulated_instance_norm_is_used(context, **kwargs):
        return False
    channels = input_x.shape[1]
    groups = kwargs["weight_for_norm"].const_value.shape[0]
    return (
        isinstance(channels, int)
        and channels > 0
        and channels % groups == 0
        and weight_full.shape[0] == bias_full.shape[0] == channels
    )


def onnx_group_normalization(
    op, input_x, weight_for_norm, weight_full, bias_full, epsilon, **_
):
    reshape_to_1d = op.Constant(value_ints=[-1])
    weight_full = op.Reshape(weight_full, reshape_to_1d)
    bias_full = op.Reshape(bias_full, reshape_to_1d)
    return op.GroupNormalization(
        input_x,
        weight_full,
        bias_full,
        num_groups=weight_for_norm.const_value.shape[0],
        epsilon=epsilon,
    )


instance_norm_to_onnx_group_norm_rule = RewriteRule(
    instance_simulates_group_normalization_pattern,
    onnx_group_normalization,
    check_onnx_group_normalization,
)

rules = [instance_norm_to_onnx_group_norm_rule]
