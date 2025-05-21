# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Sequence, TypeVar, Union

__all__ = [
    "pattern",
    "rewrite",
    "RewritePass",
]

import onnx

import onnxscript.ir.passes.common as common_passes
from onnxscript import ir
from onnxscript.rewriter import (
    broadcast_to_matmul,
    cast_constant_of_shape,
    collapse_slices,
    gemm_to_matmul_add,
    llama_rule_sets,
    no_op,
    pattern,
)

_ModelProtoOrIr = TypeVar("_ModelProtoOrIr", onnx.ModelProto, ir.Model)
_DEFAULT_REWRITE_RULES: tuple[pattern.RewriteRule, ...] = (
    *no_op.rules.rules,  # TODO: merge this rule into constant folding?
    *broadcast_to_matmul.rules.rules,
    gemm_to_matmul_add.rule,  # type: ignore[has-type]
    *cast_constant_of_shape.rules.rules,
    *collapse_slices.rules.rules,
    *llama_rule_sets.llama_p0_rule_set().rules,
)


class RewritePass(ir.passes.InPlacePass):
    def __init__(
        self,
        rules: Sequence[pattern.RewriteRule] | pattern.RewriteRuleSet,
        /,
    ) -> None:
        super().__init__()
        if isinstance(rules, Sequence):
            if not rules:
                raise ValueError("rules must not be empty")
            # Create a pattern rule-set using provided rules
            rules = pattern.RewriteRuleSet(rules)
        assert isinstance(rules, pattern.RewriteRuleSet)
        self.rules: pattern.RewriteRuleSet = rules

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        count = self.rules.apply_to_model(model)
        if count:
            print(f"Applied {count} of general pattern rewrite rules.")
        return ir.passes.PassResult(model, bool(count))


def rewrite(
    model: _ModelProtoOrIr,
    pattern_rewrite_rules: Union[Sequence[pattern.RewriteRule], pattern.RewriteRuleSet]
    | None = None,
) -> _ModelProtoOrIr:
    """Rewrite the model using the provided pattern rewrite rules.

    Unused nodes, functions, and opsets will be removed after the rewrite.

    Args:
        model: The model to be rewritten. Can be an ONNX ModelProto or an ir.Model.
        pattern_rewrite_rules: A sequence of pattern rewrite rules or a RewriteRuleSet.
            If not provided, default rules will be applied. If empty, no rules will be applied
            and the original model will be returned.

    Returns:
        The rewritten model as the same type as the input model.
    """
    if pattern_rewrite_rules is None:
        pattern_rewrite_rules = _DEFAULT_REWRITE_RULES
    elif not pattern_rewrite_rules:
        return model

    if isinstance(model, onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(model)
        proto = True
    else:
        model_ir = model
        proto = False

    rewrite_pass = ir.passes.PassManager(
        (
            RewritePass(pattern_rewrite_rules),
            common_passes.RemoveUnusedNodesPass(),
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
        )
    )
    model_ir = rewrite_pass(model_ir).model
    if proto:
        return ir.serde.serialize_model(model_ir)
    return model_ir  # type: ignore[return-value]
