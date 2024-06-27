# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

from onnxscript.rewriter import function_rule, pattern
from onnxscript.rewriter import rewrite as _rewrite
from onnxscript.rewriter.onnxruntime import (
    fused_matmul_rule_sets,
    group_normalization_merge_silu,
    instance_to_group_normalization,
    softmax,
    transformers,
)

ORT_FUNCTION_REWRITE_RULES = [*transformers.TRANSFORMERS_FUNCTION_REWRITE_RULES]

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
    # NOTE: group normalization merge silu should be applied after instance to group normalization
    *group_normalization_merge_silu.rules.rules,
    *fused_matmul_rule_sets.fused_matmul_rule_sets(),
]


def rewrite(
    model_proto: onnx.ModelProto,
    /,
    function_rules: list[type[function_rule.FunctionRewriteRule]] | None = None,
    pattern_rules: list[pattern.RewriteRule] | None = None,
) -> onnx.ModelProto:
    """Rewrite the model using the given rules.

    Args:
        model_proto: The model to rewrite.
        function_rules: The function rewrite rules to apply. If None, the default rules
            for onnxruntime are used.
        pattern_rules: The pattern rewrite rules to apply. If None, the default rules
            for onnxruntime are used.

    Returns:
        The rewritten model.
    """
    function_rules = function_rules or ORT_FUNCTION_REWRITE_RULES
    pattern_rules = pattern_rules or ORT_PATTERN_REWRITE_RULES
    return _rewrite(
        model_proto, function_rewrite_rules=function_rules, pattern_rewrite_rules=pattern_rules
    )
