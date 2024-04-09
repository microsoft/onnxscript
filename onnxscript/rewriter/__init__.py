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

from onnxscript.ir import serde
from onnxscript.optimizer import remove_unused, remove_unused_function
from onnxscript.rewriter import function_rule, pattern

PatternRewriteRule = pattern.RewriteRule
FunctionRewriteRule = function_rule.FunctionRewriteRule


def rewrite(
    model: onnx.ModelProto,
    function_rewrite_rules: Sequence[type[FunctionRewriteRule]] = (),
    pattern_rewrite_rules: Sequence[PatternRewriteRule] = (),
) -> onnx.ModelProto:
    if function_rewrite_rules:
        model_ir = serde.deserialize_model(model)
        for rule_cls in function_rewrite_rules:
            rule_cls().apply_to_model(model_ir)
        # TODO: Avoid serializing and deserializing the model?
        model = serde.serialize_model(model_ir)
    if pattern_rewrite_rules:
        model_ir = serde.deserialize_model(model)
        count = pattern.RewriteRuleSet(pattern_rewrite_rules).apply_to_model(model_ir)
        print(f"Applied {count} of general pattern rewrite rules.")
        model = serde.serialize_model(model_ir)
    remove_unused.remove_unused_nodes(model)
    remove_unused_function.remove_unused_functions(model)
    return model
