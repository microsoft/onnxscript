# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the TopologicalSortPass."""

import unittest

from onnxscript import ir
from onnxscript.ir.passes.common import topological_sort


class TopologicalSortPassTest(unittest.TestCase):
    def setUp(self):
        self.node_a = ir.node("A", inputs=[], name="node_a")
        self.node_b = ir.node("B", inputs=self.node_a.outputs, name="node_b")
        self.node_c = ir.node("C", inputs=self.node_b.outputs, name="node_c")

    def test_topological_sort_modified_true(self):
        graph = ir.Graph(
            inputs=self.node_a.inputs,
            outputs=self.node_c.outputs,
            nodes=[self.node_c, self.node_b, self.node_a],  # Unsorted nodes
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)
        result = topological_sort.TopologicalSortPass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(
            tuple(result.model.graph),
            (self.node_a, self.node_b, self.node_c),
        )

    def test_topological_sort_modified_false(self):
        """Test that modified is False when the input model is already sorted."""
        sorted_graph = ir.Graph(
            inputs=self.node_a.inputs,
            outputs=self.node_c.outputs,
            nodes=[self.node_a, self.node_b, self.node_c],  # Sorted nodes
            name="test_graph",
        )
        sorted_model = ir.Model(sorted_graph, ir_version=10)
        result = topological_sort.TopologicalSortPass()(sorted_model)
        self.assertFalse(result.modified)
        self.assertEqual(
            tuple(result.model.graph),
            (self.node_a, self.node_b, self.node_c),
        )

    def test_topological_sort_on_functions(self):
        """Test that TopologicalSortPass works on functions in a model."""
        # Create a function with unsorted nodes
        func_graph = ir.Graph(
            inputs=self.node_a.inputs,
            outputs=self.node_c.outputs,
            nodes=[self.node_c, self.node_b, self.node_a],  # Unsorted nodes
        )
        function = ir.Function(
            domain="test_domain",
            name="test_function",
            graph=func_graph,
            attributes=[],
        )

        # Create a model with the function
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10, functions=[function])

        # Apply the TopologicalSortPass
        result = topological_sort.TopologicalSortPass()(model)

        # Verify that the nodes in the function are sorted
        sorted_func_nodes = (self.node_a, self.node_b, self.node_c)
        self.assertTrue(result.modified)
        self.assertEqual(
            tuple(result.model.functions[function.identifier()]),
            sorted_func_nodes,
        )


if __name__ == "__main__":
    unittest.main()
