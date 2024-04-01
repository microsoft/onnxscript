from __future__ import annotations

from typing import Sequence

__all__ = [
    # Modules
    "irbuilder",
    "protobuilder",
    "function_rule",
    "pattern",
    # Functions
    "rewrite",
]

import onnx

from onnxscript._legacy_ir import irbuilder, protobuilder
from onnxscript.rewriter import function_rule, pattern

PatternRewriteRule = pattern.RewriteRule
FunctionRewriteRule = function_rule.FunctionRewriteRule


def rewrite(
    model: onnx.ModelProto,
    function_rewrite_rules: Sequence[type[FunctionRewriteRule]] = (),
    pattern_rewrite_rules: Sequence[PatternRewriteRule] = (),
) -> onnx.ModelProto:
    if function_rewrite_rules:
        model_ir = irbuilder.build_ir(model)
        for rule_cls in function_rewrite_rules:
            rule_cls().apply_to_model(model_ir)
        model = model_ir.original_model_proto
    if pattern_rewrite_rules:
        model_ir = irbuilder.build_ir(model)
        count = pattern.RewriteRuleSet(pattern_rewrite_rules).apply_to_model(model_ir)
        print(f"Applied {count} pattern rewrite rules.")
        model = protobuilder.build_model_proto(model_ir)
    return model
