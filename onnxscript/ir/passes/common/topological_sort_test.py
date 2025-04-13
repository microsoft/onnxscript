# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the TopologicalSortPass."""

import unittest

from onnxscript import ir
from onnxscript.ir.passes.common import topological_sort


class TopologicalSortPassTest(unittest.TestCase):
    def setUp(self):
        self.node_a = ir.Node("", "A", inputs=[], num_outputs=1, name="node_a")
        self.node_b = ir.Node("", "B", inputs=[self.node_a.outputs[0]], num_outputs=1, name="node_b")
        self.node_c = ir.Node("", "C", inputs=[self.node_b.outputs[0]], num_outputs=1, name="node_c")

    def test_topological_sort_modified_true(self):
        graph = ir.Graph(
            inputs=self.node_a.inputs,
            outputs=self.node_c.outputs,
            nodes=[self.node_c, self.node_b, self.node_a],  # Unsorted nodes
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)
        pass_result = topological_sort.TopologicalSortPass(model)
        self.assertTrue(pass_result.modified)

    def test_topological_sort_modified_false(self):
        """Test that modified is False when the input model is already sorted."""
        sorted_graph = ir.Graph(
            inputs=self.node_a.inputs,
            outputs=self.node_c.outputs,
            nodes=[self.node_a, self.node_b, self.node_c],  # Sorted nodes
            name="test_graph",
        )
        sorted_model = ir.Model(sorted_graph, ir_version=10)
        pass_result = topological_sort.TopologicalSortPass().call(sorted_model)
        self.assertFalse(pass_result.modified)


if __name__ == "__main__":
    unittest.main()
