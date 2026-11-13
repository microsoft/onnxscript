# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet
from onnxscript.rewriter.rules.common._group_normalization import (
    check_if_simulated_instance_norm_is_used,
    instance_simulates_group_normalization_pattern,
)


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
        _domain="com.microsoft",
    )
    return op.Transpose(output, perm=[0, 3, 1, 2])


# Register the rewrite rules
instance_norm_to_group_norm_rule = RewriteRule(
    instance_simulates_group_normalization_pattern,
    group_normalization,
    check_if_simulated_instance_norm_is_used,
)

# NOTE: instance_norm_to_group_norm_rule is subset of instance_norm_to_group_norm_with_silu_rule,
# so we need to run instance_norm_to_group_norm_with_silu_rule first.
rules = RewriteRuleSet([instance_norm_to_group_norm_rule])
