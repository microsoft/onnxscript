# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pattern-based rewriter API for ONNX models.

This module provides the main user-facing API for the ONNX pattern rewriter.
It allows users to define patterns that match subgraphs in ONNX models and
replace them with more efficient implementations.

Recommended Usage with Class-Based Rules:
    
    ```python
    from onnxscript.rewriter import pattern
    
    class AddZeroElimination(pattern.RewriteRuleClassBase):
        \"\"\"Removes addition with zero: Add(x, 0) -> Identity(x)\"\"\"
        
        def pattern(self, op, x):
            zero = op.Constant(value=0.0)
            return op.Add(x, zero)
        
        def check(self, context, x, zero):
            # Optional: Add conditions for when to apply this rule
            return zero.const_value is not None and zero.const_value.item() == 0.0
        
        def rewrite(self, op, x, zero=None):
            return op.Identity(x)
    
    # Create and apply the rule
    rule = AddZeroElimination.rule()
    rule.apply_to_model(model)
    ```
    
    Multiple pattern example:
    
    ```python
    class TransposeElimination(pattern.RewriteRuleClassBase):
        \"\"\"Removes redundant transpose: Transpose(Transpose(x, perm), reverse_perm) -> x\"\"\"
        
        def pattern(self, op, x, perm):
            return op.Transpose(x, perm=perm)
        
        def check(self, context, x, perm):
            # Only apply if permutation is identity (no-op transpose)
            if perm.is_ref():
                return False
            if perm.type == ir.AttributeType.INTS:
                perm_list = perm.as_ints()
                return perm_list == list(range(len(perm_list)))
            return False
        
        def rewrite(self, op, x, perm=None):
            return op.Identity(x)
    
    # Apply multiple rules as a set
    rules = pattern.RewriteRuleSet([
        AddZeroElimination.rule(),
        TransposeElimination.rule()
    ])
    rules.apply_to_model(model)
    ```
    
    Function-based rules (lower-level API):
    
    ```python
    # For simple cases, you can still use function-based rules
    def mul_one_pattern(op, x):
        one = op.Constant(value=1.0)
        return op.Mul(x, one)
    
    def identity_replacement(op, x):
        return op.Identity(x)
    
    rule = pattern.RewriteRule(mul_one_pattern, identity_replacement)
    ```

Classes and functions exported:
    - RewriteRuleClassBase: Recommended base class for implementing rewrite rules
    - RewriteRule: Core class for defining pattern-to-replacement rules
    - RewriteRuleSet: Collection of rules with application logic  
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
