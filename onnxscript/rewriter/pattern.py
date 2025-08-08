# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.ir import _tape
from onnxscript.rewriter._basics import MatchingTracer, MatchResult, MatchStatus
from onnxscript.rewriter._matcher import PatternMatcher, SimplePatternMatcher
from onnxscript.rewriter._pattern_ir import (
    ANY_VALUE,
    AttrVar,
    Constant,
    OpsetPatternBuilder,
    OrValue,
    Var,
    pattern_builder,
    torch_module_op,
)
from onnxscript.rewriter._rewrite_rule import (
    Pattern,
    PatternBase,
    RewriteRule,
    RewriteRuleClassBase,
    RewriteRuleSet,
)

RewriterContext = _tape.Builder

__all__ = [
    "ANY_VALUE",
    "AttrVar",
    "OrValue",
    "Constant",
    "OpsetPatternBuilder",
    "pattern_builder",
    "PatternBase",
    "Pattern",
    "RewriteRule",
    "RewriteRuleClassBase",
    "RewriteRuleSet",
    "RewriterContext",
    "MatchingTracer",
    "MatchResult",
    "MatchStatus",
    "PatternMatcher",
    "SimplePatternMatcher",
    "torch_module_op",
    "Var",
]
