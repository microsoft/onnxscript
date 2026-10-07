# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

from onnxscript.rewriter import _pattern_ir


class PatternIRTest(unittest.TestCase):
    """Test _pattern_ir module functionality."""

    def test_value_pattern_with_check(self):
        """Test ValuePattern with check attribute."""

        def value_checker(context, value):
            return True

        # Test creating ValuePattern with check
        value_pattern = _pattern_ir.ValuePattern("test_value", check=value_checker)
        self.assertIs(value_pattern._check, value_checker)
        self.assertEqual(value_pattern.name, "test_value")

    def test_node_pattern_with_check(self):
        """Test NodePattern with check attribute."""

        def node_checker(context, node):
            return True

        # Test creating NodePattern with check
        domain_pattern = _pattern_ir.StringConstantPattern("")
        inputs = []
        attributes = {}
        outputs = ["output"]

        node_pattern = _pattern_ir.NodePattern(
            domain_pattern,
            "Add",
            inputs,
            attributes,
            outputs,
            allow_other_attributes=True,
            allow_other_inputs=True,
            check=node_checker,
        )
        self.assertIs(node_pattern._check, node_checker)

    def test_to_value_pattern_with_callable(self):
        """Test _to_value_pattern function with callable input."""

        def my_checker(context, value):
            return True

        result = _pattern_ir._to_value_pattern(my_checker)
        self.assertIsInstance(result, _pattern_ir.ValuePattern)
        self.assertIs(result._check, my_checker)
        self.assertIsNone(result.name)

    def test_op_pattern_builder_with_check(self):
        """Test OpPatternBuilder with _check parameter."""

        def node_checker(context, node):
            return True

        # Create OpPatternBuilder and call with _check parameter
        opset_builder = _pattern_ir.OpsetPatternBuilder("")
        result = opset_builder.Add(None, None, _check=node_checker)

        # The result should be a NodeOutputPattern, and its producer should have the check
        self.assertTrue(hasattr(result, "producer"))
        producer = result.producer()
        self.assertIsNotNone(producer)
        self.assertTrue(hasattr(producer, "_check"))
        self.assertIs(producer._check, node_checker)

    def test_graph_pattern_output_nodes_have_deterministic_order(self):
        """Test that GraphPattern.output_nodes preserves insertion order from outputs.

        Regression test for https://github.com/microsoft/onnxscript/issues/2234.
        When output_nodes was built from a set, Python's hash randomization could
        cause non-deterministic ordering, leading to non-deterministic pattern
        matching behavior for multi-output patterns.
        """
        opset_builder = _pattern_ir.OpsetPatternBuilder("")
        x = _pattern_ir.ValuePattern("x")
        # Create two distinct node patterns via two separate ops
        out_a = opset_builder.Relu(x, _outputs=["a"])
        out_b = opset_builder.Sigmoid(x, _outputs=["b"])
        outputs = [out_a, out_b]

        # Build the graph pattern multiple times and check the order is always the same
        for _ in range(50):
            graph_pattern = _pattern_ir.GraphPattern(inputs=[x], outputs=outputs, nodes=[])
            node_op_ids = [n._op_identifier for n in graph_pattern.output_nodes]
            self.assertEqual(
                node_op_ids,
                [("", "Relu", ""), ("", "Sigmoid", "")],
                "output_nodes order must match the order of outputs",
            )


if __name__ == "__main__":
    unittest.main()
