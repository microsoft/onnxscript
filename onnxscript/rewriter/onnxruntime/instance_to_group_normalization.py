# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import numpy as np
import onnx

from onnxscript.rewriter import _ir_utils, pattern

torch_module_op = pattern.torch_module_op

logger = logging.getLogger(__name__)


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
    """Check if the simulated instance normalization is used.

    In torchlib with opset18, onnx.GroupNorm is using wrong definition, so
    we use InstanceNormalization to simulate GroupNormalization. We need to check if there are arguments created to simulation.
    If there are, then we need to replace the pattern. If they are not used, then we don't need to replace the pattern.

    To validate this, we need to check the following:
    1. weight_for_norm are all 1 and bias_for_norm are all 0, as they are created for the simulation.
    2. weight_full and bias_full are unsqueezed to be easily broadcastable.
    3. input rank should be 4
    4. weight_full and bias_full should have ones except first dim.
    5. adjusted_input_shape is a constant tensor of form [0, g, -1]
    6. original_input_shape is the same as input_x shape.

    Returns:
        bool: True if the simulated instance normalization is used, False otherwise.
    """
    weight_for_norm_prop = _ir_utils.propagate_const_value(weight_for_norm)
    weight_for_norm_const_value = weight_for_norm_prop.const_value
    if weight_for_norm_const_value is None:
        return False
    weight_for_norm = weight_for_norm_const_value.numpy()

    bias_for_norm_prop = _ir_utils.propagate_const_value(bias_for_norm)
    bias_for_norm_const_value = bias_for_norm_prop.const_value
    if bias_for_norm_const_value is None:
        return False
    bias_for_norm = bias_for_norm_const_value.numpy()

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

    adjusted_input_shape = _ir_utils.propagate_const_value(adjusted_input_shape)
    adjusted_input_shape_const_value = adjusted_input_shape.const_value

    g = weight_for_norm.shape[0]
    if (
        adjusted_input_shape_const_value is None
        or adjusted_input_shape_const_value.numpy().tolist() != [0, g, -1]
    ):
        return False

    # NOTE: Restrict the rule to only support constant shape
    original_input_shape = _ir_utils.propagate_const_value(original_input_shape)
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


def group_normalization(op, input_x, weight_for_norm, weight_full, bias_full, epsilon, **_):
    # com.microsoft.GroupNorm only supports NHWC for now
    nhwc_input = op.Transpose(input_x, perm=[0, 2, 3, 1])
    # com.microsoft.GroupNorm only supports gamma and beta as float type
    weight_full = op.Cast(weight_full, to=onnx.TensorProto.FLOAT)
    reshape_to_1d = op.Constant(value_ints=[-1])
    weight_full = op.Reshape(weight_full, reshape_to_1d)
    bias_full = op.Cast(bias_full, to=onnx.TensorProto.FLOAT)
    bias_full = op.Reshape(bias_full, reshape_to_1d)
    # re-obtain attribute groups
    # TODO(rama): Earlier check implies weight_for_norm is a constant tensor?
    # If not, we should add a check that shape[0] is not symbolic.
    shape = weight_for_norm.shape
    if shape is None:
        raise ValueError("weight_for_norm shape not known")
    groups = shape[0]
    output = op.GroupNorm(
        nhwc_input,
        weight_full,
        bias_full,
        activation=0,
        channels_last=1,
        epsilon=epsilon,
        groups=groups,
        domain="com.microsoft",
    )
    return op.Transpose(output, perm=[0, 3, 1, 2])


# Register the rewrite rules
instance_norm_to_group_norm_rule = pattern.RewriteRule(
    instance_simulates_group_normalization_pattern,
    group_normalization,
    check_if_simulated_instance_norm_is_used,
)

# NOTE: instance_norm_to_group_norm_rule is subset of instance_norm_to_group_norm_with_silu_rule,
# so we need to run instance_norm_to_group_norm_with_silu_rule first.
rules = pattern.RewriteRuleSet([instance_norm_to_group_norm_rule])
