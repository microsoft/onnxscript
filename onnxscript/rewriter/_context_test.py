# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for _context.py: RewriterContext ABC and TapeRewriterContext."""

from __future__ import annotations

import unittest

from onnxscript import ir
from onnxscript.rewriter._context import TapeRewriterContext


class TapeRewriterContextStorageTest(unittest.TestCase):
    """Tests for the storage/harvesting interface of TapeRewriterContext."""

    def test_add_node_and_harvest(self):
        ctx = TapeRewriterContext()
        node = ir.Node("", "Relu", [ir.Value()], num_outputs=1)
        ctx._add_node(node)
        self.assertEqual(len(ctx.nodes), 1)
        self.assertIs(ctx.nodes[0], node)

    def test_add_initializer_and_harvest(self):
        ctx = TapeRewriterContext()
        tensor = ir.tensor([1.0, 2.0], name="init")
        value = ir.Value(name="init", const_value=tensor)
        ctx._add_initializer(value)
        self.assertEqual(len(ctx.initializers), 1)
        self.assertIs(ctx.initializers[0], value)

    def test_record_opset(self):
        ctx = TapeRewriterContext()
        ctx._record_opset("", 20)
        ctx._record_opset("com.microsoft", 1)
        self.assertEqual(ctx.used_opsets, {("", 20), ("com.microsoft", 1)})

    def test_empty_context(self):
        ctx = TapeRewriterContext()
        self.assertEqual(len(ctx.nodes), 0)
        self.assertEqual(len(ctx.initializers), 0)
        self.assertEqual(len(ctx.used_opsets), 0)


class RewriterContextOpCreationTest(unittest.TestCase):
    """Tests that op creation works correctly via TapeRewriterContext."""

    def test_dynamic_dispatch_single_output(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        result = op.Relu(x)
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(len(op.nodes), 1)
        self.assertEqual(op.nodes[0].op_type, "Relu")

    def test_dynamic_dispatch_with_domain(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        result = op.BiasGelu(x, _domain="com.microsoft")
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(op.nodes[0].domain, "com.microsoft")
        self.assertIn(("com.microsoft", None), op.used_opsets)

    def test_dynamic_dispatch_multi_output(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        results = op.Split(x, _outputs=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(v, ir.Value) for v in results))
        self.assertEqual(len(op.nodes), 1)

    def test_dynamic_dispatch_named_outputs(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        results = op.Split(x, _outputs=["a", "b"])
        self.assertEqual(results[0].name, "a")
        self.assertEqual(results[1].name, "b")

    def test_dynamic_dispatch_with_name(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        _ = op.Relu(x, _name="my_relu")
        self.assertEqual(op.nodes[0].name, "my_relu")

    def test_op_method_explicit(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        w = ir.Value(name="w")
        result = op.op("Conv", inputs=[x, w], domain="", name="my_conv")
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(op.nodes[0].op_type, "Conv")
        self.assertEqual(op.nodes[0].name, "my_conv")

    def test_op_method_with_attributes(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        result = op.op("Elu", inputs=[x], attributes={"alpha": 2.0})
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(op.nodes[0].op_type, "Elu")
        self.assertIn("alpha", op.nodes[0].attributes)

    def test_op_method_with_attr_map(self):
        """Verify that passing node.attributes (an Attributes mapping) works."""
        op = TapeRewriterContext()
        source_node = ir.Node(
            "",
            "Conv",
            [ir.Value()],
            attributes=[ir.AttrInt64s("pads", [1, 1, 1, 1])],
            num_outputs=1,
        )
        result = op.op("Conv", inputs=[ir.Value()], attributes=source_node.attributes)
        self.assertIsInstance(result, ir.Value)
        self.assertIn("pads", op.nodes[0].attributes)

    def test_initializer_creation(self):
        op = TapeRewriterContext()
        tensor = ir.tensor([1.0, 2.0, 3.0], name="my_init")
        value = op.initializer(tensor, name="my_init")
        self.assertIsInstance(value, ir.Value)
        self.assertEqual(value.name, "my_init")
        self.assertEqual(len(op.initializers), 1)

    def test_initializer_requires_name(self):
        op = TapeRewriterContext()
        tensor = ir.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            op.initializer(tensor)

    def test_opset_recording(self):
        op = TapeRewriterContext()
        x = ir.Value(name="x")
        _ = op.Relu(x)
        _ = op.BiasGelu(x, _domain="com.microsoft", _version=1)
        self.assertIn(("", None), op.used_opsets)
        self.assertIn(("com.microsoft", 1), op.used_opsets)


if __name__ == "__main__":
    unittest.main()
