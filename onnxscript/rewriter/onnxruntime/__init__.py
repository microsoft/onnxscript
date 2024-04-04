from __future__ import annotations

import onnx

from onnxscript._legacy_ir import irbuilder, protobuilder
from onnxscript.optimizer import remove_unused, remove_unused_function
from onnxscript.rewriter import function_rule, pattern
from onnxscript.rewriter.onnxruntime import (
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
]


def rewrite(
    model: onnx.ModelProto,
    function_rules: list[type[function_rule.FunctionRewriteRule]] | None = None,
    pattern_rules: list[pattern.RewriteRule] | None = None,
) -> onnx.ModelProto:
    """Rewrite the model using the given rules.

    Args:
        model: The model to rewrite.
        function_rules: The function rewrite rules to apply. If None, the default rules
            for onnxruntime are used.
        pattern_rules: The pattern rewrite rules to apply. If None, the default rules
            for onnxruntime are used.

    Returns:
        The rewritten model.
    """
    function_rules = function_rules or ORT_FUNCTION_REWRITE_RULES
    pattern_rules = pattern_rules or ORT_PATTERN_REWRITE_RULES
    # TODO: Function rules first, or pattern rules first?
    if function_rules:
        model_ir = irbuilder.build_ir(model)
        for rule_cls in function_rules:
            rule_cls().apply_to_model(model_ir)
        model = model_ir.original_model_proto
    if pattern_rules:
        model_ir = irbuilder.build_ir(model)
        count = pattern.RewriteRuleSet(pattern_rules).apply_to_model(model_ir)
        print(f"Applied {count} pattern rewrite rules.")
        model = protobuilder.build_model_proto(model_ir)
    # TODO: Does it make more sense we run DCE after each rewrite rule applied?
    # If so, we need IR to support DCE.
    remove_unused.remove_unused_nodes(model)
    remove_unused_function.remove_unused_functions(model)
    return model
