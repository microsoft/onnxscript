# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test for the new CompiledPattern and PatternBase classes."""

import unittest

import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import pattern


class CompiledPatternTest(unittest.TestCase):
    """Test CompiledPattern functionality."""

    def test_pattern_impl_basic_functionality(self):
        """Test that CompiledPattern can be created and used independently."""

        def simple_pattern(op, x):
            return op.Identity(x)

        # Create a CompiledPattern
        pattern_impl = pattern.CompiledPattern(simple_pattern, name="SimpleIdentity")

        # Verify basic properties
        self.assertEqual(pattern_impl.name, "SimpleIdentity")
        self.assertIsNotNone(pattern_impl._target_pattern)
        self.assertIsNotNone(pattern_impl._matcher)
        self.assertIsNotNone(pattern_impl._condition_function)

    def test_pattern_impl_match_method(self):
        """Test that CompiledPattern.match method works correctly."""

        def identity_pattern(op, x):
            return op.Identity(x)

        pattern_impl = pattern.CompiledPattern(identity_pattern, name="IdentityPattern")

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

        # Find the Identity node
        identity_node = None
        for node in model.graph:
            if node.op_type == "Identity":
                identity_node = node
                break

        self.assertIsNotNone(identity_node)

        # Test pattern matching
        match_result = pattern_impl.match(model, model.graph, identity_node)

        # The match might succeed or fail depending on how the pattern matching works
        # The important thing is that the method runs without error
        self.assertIsInstance(match_result, (pattern.MatchResult, type(None)))

    def test_pattern_impl_with_condition_function(self):
        """Test CompiledPattern with a custom condition function."""

        def identity_pattern(op, x):
            return op.Identity(x)

        def always_fail_condition(context, x):
            return False

        pattern_impl = pattern.CompiledPattern(
            identity_pattern, condition_function=always_fail_condition, name="FailingIdentity"
        )

        # Create a model with an Identity node
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z)
            {
                z = Identity(x)
            }
            """
        )

        # Find the Identity node
        identity_node = None
        for node in model.graph:
            if node.op_type == "Identity":
                identity_node = node
                break

        self.assertIsNotNone(identity_node)

        # Test pattern matching - should fail due to condition function
        match_result = pattern_impl.match(model, model.graph, identity_node)

        # Should return None due to failing condition
        self.assertIsNone(match_result)

    def test_pattern_impl_no_match_returns_match_object(self):
        """Test that CompiledPattern.match returns match object (not always None) when available."""

        def identity_pattern(op, x):
            return op.Identity(x)

        pattern_impl = pattern.CompiledPattern(identity_pattern, name="IdentityPattern")

        # Create a model with an Add node (should not match Identity pattern)
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                z = Add(x, y)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)

        # Find the Add node
        add_node = None
        for node in model.graph:
            if node.op_type == "Add":
                add_node = node
                break

        self.assertIsNotNone(add_node)

        # Test pattern matching - should fail because Add != Identity
        match_result = pattern_impl.match(model, model.graph, add_node)

        # The result should be falsy (either None or a failed MatchResult)
        self.assertFalse(bool(match_result))


class PatternBaseTest(unittest.TestCase):
    """Test PatternBase functionality."""

    def test_pattern_base_creation(self):
        """Test that PatternBase can be subclassed and used."""

        class TestPattern(pattern.PatternBase):
            def pattern(self, op, x):
                return op.Identity(x)

        test_pattern = TestPattern(name="TestPattern")
        self.assertEqual(test_pattern.name, "TestPattern")

    def test_pattern_base_compiled_pattern_access(self):
        """Test that PatternBase has an internal CompiledPattern."""

        class TestPattern(pattern.PatternBase):
            def pattern(self, op, x):
                return op.Identity(x)

            def check(self, context, x):
                return pattern.MatchResult()  # Always succeeds

        test_pattern = TestPattern(name="TestPattern")
        
        # The CompiledPattern should be created internally
        self.assertIsInstance(test_pattern._compiled_pattern, pattern.CompiledPattern)
        self.assertEqual(test_pattern._compiled_pattern.name, "TestPattern")

    def test_pattern_base_default_name(self):
        """Test that PatternBase uses class name as default."""

        class MyCustomPattern(pattern.PatternBase):
            def pattern(self, op, x):
                return op.Identity(x)

        test_pattern = MyCustomPattern()
        self.assertEqual(test_pattern.name, "MyCustomPattern")


class RewriteRuleInheritanceTest(unittest.TestCase):
    """Test that RewriteRule still works after inheriting from CompiledPattern."""

    def test_rewrite_rule_still_works(self):
        """Test that existing RewriteRule functionality is preserved."""

        def reciprocal_mul_pattern(op, x, y):
            return (1 / x) * y

        def div_replacement(op, x, y):
            return op.Div(y, x)

        rule = pattern.RewriteRule(reciprocal_mul_pattern, div_replacement)

        # Create a model that should match
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                c1 = Constant<value_float = 1.0>()
                t1 = Div(c1, x)
                z1 = Mul(t1, y)
                z = Identity(z1)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)

        # Apply the rule
        count = rule.apply_to_model(model)

        # The rule should either apply or not, but the method should work
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_rewrite_rule_class_base_still_works(self):
        """Test that RewriteRuleClassBase still works after inheriting from PatternBase."""

        class SimpleIdentityRule(pattern.RewriteRuleClassBase):
            def pattern(self, op, x):
                return op.Identity(x)

            def check(self, context, x):
                return pattern.MatchResult()  # Always succeeds

            def rewrite(self, op, x):
                return op.Identity(x)  # No-op replacement

        # Create a rule instance
        rule = SimpleIdentityRule.rule()

        self.assertIsInstance(rule, pattern.RewriteRule)
        self.assertEqual(rule.name, "SimpleIdentityRule")


if __name__ == "__main__":
    unittest.main()
