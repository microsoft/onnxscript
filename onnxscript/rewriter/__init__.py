from __future__ import annotations

from typing import Sequence, Union

__all__ = [
    # Modules
    "function_rule",
    "pattern",
    # Functions
    "rewrite",
]

import onnx

from onnxscript import ir
from onnxscript.optimizer import remove_unused, remove_unused_function
from onnxscript.rewriter import function_rule, pattern

RewriteRuleSet = pattern.RewriteRuleSet
PatternRewriteRule = pattern.RewriteRule
FunctionRewriteRule = function_rule.FunctionRewriteRule


def rewrite(
    model: onnx.ModelProto,
    function_rewrite_rules: Sequence[type[FunctionRewriteRule]] = (),
    pattern_rewrite_rules: Union[Sequence[PatternRewriteRule], RewriteRuleSet] = (),
) -> onnx.ModelProto:
    model_ir = ir.serde.deserialize_model(model)
    if function_rewrite_rules:
        for rule_cls in function_rewrite_rules:
            count, model_ir = rule_cls().apply_to_model(model_ir)
            print(f"Applied {count} of onnxruntime specific function rewrite rules.")
    if pattern_rewrite_rules:
        if not isinstance(pattern_rewrite_rules, RewriteRuleSet):
            # Create a pattern rule-set using provided rules
            pattern_rewrite_rules = pattern.RewriteRuleSet(pattern_rewrite_rules)
        count = pattern_rewrite_rules.apply_to_model(model_ir)
        print(f"Applied {count} of general pattern rewrite rules.")
    model = ir.serde.serialize_model(model_ir)
    remove_unused.remove_unused_nodes(model)
    model = remove_unused_function.remove_unused_functions(model)
    return model
