# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for _context.py and _node_sink.py."""

from __future__ import annotations

import unittest

from onnxscript import ir
from onnxscript.rewriter._context import RewriterContext
from onnxscript.rewriter._node_sink import TapeSink


class TapeSinkTest(unittest.TestCase):
    def test_add_node_and_harvest(self):
        sink = TapeSink()
        node = ir.Node("", "Relu", [ir.Value()], num_outputs=1)
        sink.add_node(node)
        self.assertEqual(len(sink.nodes), 1)
        self.assertIs(sink.nodes[0], node)

    def test_add_initializer_and_harvest(self):
        sink = TapeSink()
        tensor = ir.tensor([1.0, 2.0], name="init")
        value = ir.Value(name="init", const_value=tensor)
        sink.add_initializer(value)
        self.assertEqual(len(sink.initializers), 1)
        self.assertIs(sink.initializers[0], value)

    def test_record_opset(self):
        sink = TapeSink()
        sink.record_opset("", 20)
        sink.record_opset("com.microsoft", 1)
        self.assertEqual(sink.used_opsets, {("", 20), ("com.microsoft", 1)})

    def test_empty_sink(self):
        sink = TapeSink()
        self.assertEqual(len(sink.nodes), 0)
        self.assertEqual(len(sink.initializers), 0)
        self.assertEqual(len(sink.used_opsets), 0)


class RewriterContextForbiddenAccessTest(unittest.TestCase):
    """Tests that forbidden attributes raise AttributeError."""

    def _make_context(self) -> RewriterContext:
        return RewriterContext(TapeSink())

    def test_nodes_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx.nodes

    def test_initializers_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx.initializers

    def test_used_opsets_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx.used_opsets

    def test_private_nodes_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx._nodes

    def test_private_initializers_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx._initializers

    def test_private_used_opsets_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx._used_opsets

    def test_sink_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx._sink

    def test_graph_like_raises(self):
        ctx = self._make_context()
        with self.assertRaises(AttributeError):
            _ = ctx.graph_like


class RewriterContextOpCreationTest(unittest.TestCase):
    """Tests that op creation works correctly via RewriterContext."""

    def test_dynamic_dispatch_single_output(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        result = op.Relu(x)
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(len(sink.nodes), 1)
        self.assertEqual(sink.nodes[0].op_type, "Relu")

    def test_dynamic_dispatch_with_domain(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        result = op.BiasGelu(x, _domain="com.microsoft")
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(sink.nodes[0].domain, "com.microsoft")
        self.assertIn(("com.microsoft", None), sink.used_opsets)

    def test_dynamic_dispatch_multi_output(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        results = op.Split(x, _outputs=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(v, ir.Value) for v in results))
        self.assertEqual(len(sink.nodes), 1)

    def test_dynamic_dispatch_named_outputs(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        results = op.Split(x, _outputs=["a", "b"])
        self.assertEqual(results[0].name, "a")
        self.assertEqual(results[1].name, "b")

    def test_dynamic_dispatch_with_name(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        _ = op.Relu(x, _name="my_relu")
        self.assertEqual(sink.nodes[0].name, "my_relu")

    def test_op_method_explicit(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        w = ir.Value(name="w")
        result = op.op("Conv", inputs=[x, w], domain="", name="my_conv")
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(sink.nodes[0].op_type, "Conv")
        self.assertEqual(sink.nodes[0].name, "my_conv")

    def test_op_method_with_attributes(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        result = op.op("Elu", inputs=[x], attributes={"alpha": 2.0})
        self.assertIsInstance(result, ir.Value)
        self.assertEqual(sink.nodes[0].op_type, "Elu")
        self.assertIn("alpha", sink.nodes[0].attributes)

    def test_op_method_with_attr_map(self):
        """Verify that passing node.attributes (an Attributes mapping) works."""
        sink = TapeSink()
        op = RewriterContext(sink)
        # Create a node with attributes
        source_node = ir.Node(
            "",
            "Conv",
            [ir.Value()],
            attributes=[ir.AttrInt64s("pads", [1, 1, 1, 1])],
            num_outputs=1,
        )
        # Forward attributes to a new node via op.op()
        result = op.op("Conv", inputs=[ir.Value()], attributes=source_node.attributes)
        self.assertIsInstance(result, ir.Value)
        self.assertIn("pads", sink.nodes[0].attributes)

    def test_initializer_creation(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        tensor = ir.tensor([1.0, 2.0, 3.0], name="my_init")
        value = op.initializer(tensor, name="my_init")
        self.assertIsInstance(value, ir.Value)
        self.assertEqual(value.name, "my_init")
        self.assertEqual(len(sink.initializers), 1)

    def test_initializer_requires_name(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        tensor = ir.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            op.initializer(tensor)

    def test_opset_recording(self):
        sink = TapeSink()
        op = RewriterContext(sink)
        x = ir.Value(name="x")
        _ = op.Relu(x)
        _ = op.BiasGelu(x, _domain="com.microsoft", _version=1)
        self.assertIn(("", None), sink.used_opsets)
        self.assertIn(("com.microsoft", 1), sink.used_opsets)


if __name__ == "__main__":
    unittest.main()
