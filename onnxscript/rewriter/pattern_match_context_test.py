# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test for PatternMatchContext functionality."""

import unittest

import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import pattern


class PatternMatchContextTest(unittest.TestCase):
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
            self.assertIs(context.graph_or_function, model.graph)
            self.assertIs(context.main_root_node, model.graph.node[2])
            
            # Verify that we can inspect the matched nodes
            self.assertEqual(len(context.nodes), 2)
            
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