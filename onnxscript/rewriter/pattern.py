# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pattern-based rewriter API for ONNX models.

This module provides the main user-facing API for the ONNX pattern rewriter.
It allows users to define patterns that match subgraphs in ONNX models and
replace them with more efficient implementations.

Example Usage:
    Basic pattern rewriting:
    
    ```python
    from onnxscript.rewriter import pattern
    
    # Define a pattern that matches Add(x, 0)
    def add_zero_pattern(op, x):
        zero = op.Constant(value=0.0)
        return op.Add(x, zero)
    
    # Define replacement that just returns x  
    def identity_replacement(op, x):
        return op.Identity(x)
    
    # Create and apply the rule
    rule = pattern.RewriteRule(add_zero_pattern, identity_replacement)
    rule.apply_to_model(model)
    ```
    
    Pattern with condition:
    
    ```python
    def conditional_pattern(op, x, y):
        return op.Mul(x, y)
    
    def optimized_replacement(op, x, y):
        return op.Mul(y, x)  # Commute for some optimization
    
    def check_condition(context, x, y):
        # Only apply if y is a constant
        return y.const_value is not None
    
    rule = pattern.RewriteRule(
        conditional_pattern,
        optimized_replacement, 
        check_condition
    )
    ```
    
Classes and functions exported:
    - RewriteRule: Core class for defining pattern-to-replacement rules
    - RewriteRuleSet: Collection of rules with application logic
    - RewriteRuleClassBase: Base class for implementing rules as classes
    - Pattern building utilities: OpsetPatternBuilder, pattern_builder, etc.
    - Matching utilities: MatchResult, MatchingTracer, etc.
"""
from __future__ import annotations

from onnxscript.ir import _tape
from onnxscript.rewriter._basics import MatchingTracer, MatchResult, MatchStatus
from onnxscript.rewriter._matcher import PatternMatcher, SimplePatternMatcher
from onnxscript.rewriter._pattern_ir import (
    ANY_VALUE,
    Constant,
    OpsetPatternBuilder,
    OrValue,
    pattern_builder,
    torch_module_op,
)
from onnxscript.rewriter._rewrite_rule import (
    RewriteRule,
    RewriteRuleClassBase,
    RewriteRuleSet,
)

RewriterContext = _tape.Builder

__all__ = [
    "ANY_VALUE",
    "OrValue",
    "Constant",
    "OpsetPatternBuilder",
    "pattern_builder",
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
]
