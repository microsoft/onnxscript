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
from onnxscript.ir.passes.common import unused_removal
from onnxscript.rewriter import pattern

PatternRewriteRule = pattern.RewriteRule

ModelProtoOrIr = TypeVar("ModelProtoOrIr", onnx.ModelProto, ir.Model)


class RewritePass(ir.passes.InPlacePass):
    def __init__(
        self,
        pattern_rewrite_rules: Sequence[PatternRewriteRule] | pattern.RewriteRuleSet = (),
    ) -> None:
        if pattern_rewrite_rules:
            if not isinstance(pattern_rewrite_rules, pattern.RewriteRuleSet):
                # Create a pattern rule-set using provided rules
                pattern_rewrite_rules = pattern.RewriteRuleSet(pattern_rewrite_rules)
        assert isinstance(pattern_rewrite_rules, pattern.RewriteRuleSet)
        self.pattern_rewrite_rules: pattern.RewriteRuleSet = pattern_rewrite_rules

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = self.pattern_rewrite_rules.apply_to_model(model)
        if count:
            print(f"Applied {count} of general pattern rewrite rules.")
        return ir.passes.PassResult(model, bool(count))


def rewrite(
    model: ModelProtoOrIr,
    pattern_rewrite_rules: Union[Sequence[PatternRewriteRule], pattern.RewriteRuleSet] = (),
) -> ModelProtoOrIr:
    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
        proto = True
    else:
        model_ir = model
        proto = False

    rewrite_pass = ir.passes.PassManager(
        (
            RewritePass(pattern_rewrite_rules),
            unused_removal.RemoveUnusedNodesPass(),
            unused_removal.RemoveUnusedFunctionsPass(),
            unused_removal.RemoveUnusedOpsetsPass(),
        )
    )
    model_ir = rewrite_pass(model_ir).model
    if proto:
        return ir.serde.serialize_model(model_ir)
    return model_ir  # type: ignore[return-value]
