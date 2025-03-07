# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

from onnxscript.rewriter import pattern

torch_module_op = pattern.torch_module_op

logger = logging.getLogger(__name__)


def group_normalization_and_silu_submodule(
    op,
    input,
    weight,
    bias,
    epsilon,
    groups,
):
    group_norm = op.GroupNorm(
        input,
        weight,
        bias,
        activation=0,
        channels_last=1,
        epsilon=epsilon,
        groups=groups,
        _domain="com.microsoft",
    )
    transposed = op.Transpose(group_norm, perm=[0, 3, 1, 2])
    return torch_module_op.submodule("torch_nn_modules_activation_SiLU")(
        transposed
    )  # TODO(rama)


def group_normalization_with_silu(
    op,
    input,
    weight,
    bias,
    epsilon,
    groups,
):
    group_norm = op.GroupNorm(
        input,
        weight,
        bias,
        activation=1,
        channels_last=1,
        epsilon=epsilon,
        groups=groups,
        _domain="com.microsoft",
    )
    return op.Transpose(group_norm, perm=[0, 3, 1, 2])


group_normalization_merge_silu_submodule_rule = pattern.RewriteRule(
    group_normalization_and_silu_submodule,
    group_normalization_with_silu,
)

rules = pattern.RewriteRuleSet([group_normalization_merge_silu_submodule_rule])
