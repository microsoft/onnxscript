# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test for PatternMatchContext functionality."""

import unittest

import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import pattern


class PatternMatchContextTest(unittest.TestCase):
    def test_pattern_match_context_properties(self):
        """Test that PatternMatchContext provides the expected properties."""
        
        # Create a simple model with a reciprocal multiplication pattern
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                c1 = Constant<value_float = 1.0>()
                t1 = Div(c1, x)
                z = Mul(t1, y)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        
        # Store context from condition function
        captured_context = None
        
        def capture_context_condition(context, x, y):
            nonlocal captured_context
            captured_context = context
            return True  # Always match for testing
            
        def reciprocal_mul_pattern(op, x, y):
            return (1 / x) * y

        def replacement(op, x, y):
            return op.Div(y, x)

        # Create a rule with a condition that captures the context
        rule = pattern.RewriteRule(
            reciprocal_mul_pattern, 
            replacement,
            condition_function=capture_context_condition
        )
        
        # Apply the rule to trigger the condition function
        count = rule.apply_to_model(model)
        
        # Verify the rule matched
        self.assertEqual(count, 1)
        
        # Verify that context was captured
        self.assertIsNotNone(captured_context)
        
        # Test all required properties exist and have correct types
        self.assertTrue(hasattr(captured_context, 'model'))
        self.assertTrue(hasattr(captured_context, 'graph_or_function'))
        self.assertTrue(hasattr(captured_context, 'main_root_node'))
        self.assertTrue(hasattr(captured_context, 'output_values'))
        self.assertTrue(hasattr(captured_context, 'nodes'))
        
        # Test that properties return expected types
        self.assertIsInstance(captured_context.model, ir.Model)
        self.assertIsInstance(captured_context.graph_or_function, (ir.Graph, ir.Function))
        self.assertIsInstance(captured_context.main_root_node, ir.Node)
        
        # Test that nodes and output_values return sequences
        from collections.abc import Sequence
        self.assertIsInstance(captured_context.nodes, Sequence)
        self.assertIsInstance(captured_context.output_values, Sequence)
        
        # Test that the captured context has the expected data
        self.assertEqual(captured_context.model, model)
        self.assertEqual(captured_context.graph_or_function, model.graph)
        
        # The main root node should be the Mul operation (last node in the pattern)
        self.assertEqual(captured_context.main_root_node.op_type, "Mul")
        
        # Should have nodes from the matched pattern
        self.assertGreater(len(captured_context.nodes), 0)

    def test_pattern_match_context_readonly(self):
        """Test that PatternMatchContext properties are read-only."""
        
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
        
        captured_context = None
        
        def capture_context_condition(context, x):
            nonlocal captured_context
            captured_context = context
            return True
            
        def identity_pattern(op, x):
            return op.Identity(x)

        def replacement(op, x):
            return x  # Remove identity

        rule = pattern.RewriteRule(
            identity_pattern, 
            replacement,
            condition_function=capture_context_condition
        )
        
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertIsNotNone(captured_context)
        
        # The properties should be read-only (no setters)
        with self.assertRaises(AttributeError):
            captured_context.model = None
        with self.assertRaises(AttributeError):
            captured_context.graph_or_function = None
        with self.assertRaises(AttributeError):
            captured_context.main_root_node = None
        with self.assertRaises(AttributeError):
            captured_context.output_values = None
        with self.assertRaises(AttributeError):
            captured_context.nodes = None

    def test_backward_compatibility_with_no_context(self):
        """Test that condition functions that don't use context still work."""
        
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
        
        # Condition function that ignores context parameter (old style)
        def old_style_condition(context, x):
            # This should work even though context is passed as first parameter
            return True
            
        def identity_pattern(op, x):
            return op.Identity(x)

        def replacement(op, x):
            return x

        rule = pattern.RewriteRule(
            identity_pattern, 
            replacement,
            condition_function=old_style_condition
        )
        
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)

    def test_context_usage_in_condition_function(self):
        """Test that PatternMatchContext can be meaningfully used in condition functions."""
        
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] y) => (float[N] z)
            {
                c1 = Constant<value_float = 1.0>()
                t1 = Div(c1, x)
                z = Mul(t1, y)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        
        def condition_using_context(context, x, y):
            # Use context to check properties of the match
            self.assertIs(context.model, model)
            self.assertIsNotNone(context.graph_or_function)
            self.assertIsNotNone(context.main_root_node)
            
            # Verify that we can inspect the matched nodes
            self.assertGreater(len(context.nodes), 0)
            
            # Verify that the main root node is the expected operation
            self.assertEqual(context.main_root_node.op_type, "Mul")
            
            # Check that we can access the model and graph
            self.assertIsInstance(context.model, ir.Model)
            self.assertIsInstance(context.graph_or_function, ir.Graph)
            
            return True  # Allow the rewrite
            
        def reciprocal_mul_pattern(op, x, y):
            return (1 / x) * y

        def replacement(op, x, y):
            return op.Div(y, x)

        rule = pattern.RewriteRule(
            reciprocal_mul_pattern, 
            replacement,
            condition_function=condition_using_context
        )
        
        count = rule.apply_to_model(model)
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()