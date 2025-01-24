# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fusion optimizations for ORT backend."""

__all__ = [
    "optimize_for_ort",
    "transformers",
]

from onnxscript.rewriter.ort_fusions import (
    fused_matmul_rule_sets,
    # group_normalization_merge_silu,
    instance_to_group_normalization,
    softmax,
    transformers,
)
from onnxscript.rewriter.ort_fusions._core import optimize_for_ort

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
    # NOTE: group normalization merge silu should be applied after instance to group normalization
    # *group_normalization_merge_silu.rules.rules,
    *fused_matmul_rule_sets.fused_matmul_rule_sets(),
]
