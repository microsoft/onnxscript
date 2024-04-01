from __future__ import annotations

import onnx

from onnxrewriter.ir import irbuilder, protobuilder
from onnxrewriter.optimizer import remove_unused
from onnxrewriter.rewriter import function_rule, pattern
from onnxrewriter.rewriter.onnxruntime import (
    instance_to_group_normalization,
    softmax,
    transformers,
)

ORT_FUNCTION_REWRITE_RULES = [*transformers.TRANSFORMERS_FUNCTION_REWRITE_RULES]

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
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
    remove_unused.remove_unused_nodes(model)
    return model
