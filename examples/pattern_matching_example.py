# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Example demonstrating the new pattern matching functionality."""

import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import pattern


def example_standalone_pattern_matching():
    """Example showing how to use Pattern for standalone pattern matching."""

    print("=== Standalone Pattern Matching Example ===")

    # Define a pattern that matches Identity nodes
    def identity_pattern(op, x):
        return op.Identity(x)

    # Create a Pattern for standalone pattern matching (no replacement)
    pattern_matcher = pattern.Pattern(identity_pattern, name="IdentityMatcher")

    # Create a model with an Identity node
    model_proto = onnx.parser.parse_model(
        """
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[N] x) => (float[N] z)
        {
            z = Identity(x)
        }
    """
    )
    model = ir.serde.deserialize_model(model_proto)

    # Find nodes to test pattern matching against
    for node in model.graph:
        print(f"Testing pattern against {node.op_type} node...")
        match_result = pattern_matcher.match(model, model.graph, node)

        if match_result is not None:
            print(f"  ✓ Pattern matched! Found {len(match_result.nodes)} nodes in match.")
            print(f"    Matched node: {match_result.nodes[0].op_type}")
        else:
            print(f"  ✗ Pattern did not match {node.op_type} node.")


def example_class_based_pattern():
    """Example showing how to use PatternBase for class-based pattern definition."""

    print("\n=== Class-Based Pattern Example ===")

    class IdentityPatternClass(pattern.PatternBase):
        """A class-based pattern that matches Identity nodes."""

        def pattern(self, op, x):
            return op.Identity(x)

        def check(self, context, x):
            """Custom condition - always succeeds for this example."""
            print(f"    Checking condition for input: {x}")
            return pattern.MatchResult()  # Always succeeds

    # Create an instance of the pattern class
    identity_pattern_class = IdentityPatternClass(name="ClassBasedIdentity")

    # The Pattern is created internally, we can use the pattern directly
    print(f"Created pattern matcher: {identity_pattern_class.name}")

    # Use it directly with the match method
    model_proto = onnx.parser.parse_model(
        """
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[N] x) => (float[N] z)
        {
            z = Identity(x)
        }
    """
    )
    model = ir.serde.deserialize_model(model_proto)

    for node in model.graph:
        if node.op_type == "Identity":
            print(f"Testing class-based pattern against {node.op_type} node...")
            match_result = identity_pattern_class.match(model, model.graph, node)

            if match_result is not None:
                print("  ✓ Class-based pattern matched!")
            else:
                print("  ✗ Class-based pattern did not match.")


def example_rewrite_rule_still_works():
    """Example showing that existing RewriteRule functionality is preserved."""

    print("\n=== Existing RewriteRule Still Works ===")

    def identity_pattern(op, x):
        return op.Identity(x)

    def identity_replacement(op, x):
        return op.Identity(x)  # No-op replacement

    # Create a RewriteRule (which now inherits from Pattern)
    rule = pattern.RewriteRule(identity_pattern, identity_replacement, name="IdentityRule")

    print(f"Created rewrite rule: {rule.name}")
    print(f"Rule is also a Pattern: {isinstance(rule, pattern.Pattern)}")

    # The rule can be used both for pattern matching and rewriting
    model_proto = onnx.parser.parse_model(
        """
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[N] x) => (float[N] z)
        {
            z = Identity(x)
        }
    """
    )
    model = ir.serde.deserialize_model(model_proto)

    # Use it for just pattern matching (inherited from Pattern)
    for node in model.graph:
        if node.op_type == "Identity":
            print(f"Using RewriteRule for pattern matching on {node.op_type}...")
            match_result = rule.match(model, model.graph, node)

            if match_result is not None:
                print("  ✓ RewriteRule matched as a pattern matcher!")

            # Use it for rewriting (original functionality)
            print("Using RewriteRule for rewriting...")
            count = rule.apply_to_model(model)
            print(f"  Applied rule {count} times")


if __name__ == "__main__":
    example_standalone_pattern_matching()
    example_class_based_pattern()
    example_rewrite_rule_still_works()
    print("\n=== All Examples Completed ===")
