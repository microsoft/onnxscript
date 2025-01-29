# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, TypeVar, Union

__all__ = [
    # Modules
    "pattern",
    # Functions
    "rewrite",
]

import onnx

from onnxscript import ir
from onnxscript.optimizer import _remove_unused, _remove_unused_function
from onnxscript.rewriter import pattern

RewriteRuleSet = pattern.RewriteRuleSet
PatternRewriteRule = pattern.RewriteRule

ModelProtoOrIr = TypeVar("ModelProtoOrIr", onnx.ModelProto, ir.Model)


def rewrite(
    model: ModelProtoOrIr,
    pattern_rewrite_rules: Union[Sequence[PatternRewriteRule], RewriteRuleSet] = (),
) -> ModelProtoOrIr:
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
        proto = True
    else:
        model_ir = model
        proto = False
    if pattern_rewrite_rules:
        if not isinstance(pattern_rewrite_rules, RewriteRuleSet):
            # Create a pattern rule-set using provided rules
            pattern_rewrite_rules = pattern.RewriteRuleSet(pattern_rewrite_rules)
        count = pattern_rewrite_rules.apply_to_model(model_ir)
        if count:
            print(f"Applied {count} of general pattern rewrite rules.")
    _remove_unused.remove_unused_nodes(model_ir)
    model_ir = _remove_unused_function.remove_unused_functions(model_ir)
    if proto:
        model = ir.serde.serialize_model(model_ir)
        return model
    return model_ir  # type: ignore[return-value]
