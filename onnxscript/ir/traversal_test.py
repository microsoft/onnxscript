# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import parameterized

from onnxscript import ir
from onnxscript.ir import traversal


class RecursiveGraphIteratorTest(unittest.TestCase):
    def setUp(self):
        self.graph = ir.Graph(
            [],
            [],
            nodes=[
                ir.Node("", "Node1", []),
                ir.Node("", "Node2", []),
                ir.Node(
                    "",
                    "If",
                    [],
                    attributes=[
                        ir.AttrGraph(
                            "then_branch",
                            ir.Graph(
                                [],
                                [],
                                nodes=[ir.Node("", "Node3", []), ir.Node("", "Node4", [])],
                                name="then_graph",
                            ),
                        ),
                        ir.AttrGraph(
                            "else_branch",
                            ir.Graph(
                                [],
                                [],
                                nodes=[ir.Node("", "Node5", []), ir.Node("", "Node6", [])],
                                name="else_graph",
                            ),
                        ),
                    ],
                ),
            ],
            name="main_graph",
        )

    @parameterized.parameterized.expand(
        [
            ("forward", False, ("Node1", "Node2", "If", "Node3", "Node4", "Node5", "Node6")),
            ("reversed", True, ("If", "Node4", "Node3", "Node6", "Node5", "Node2", "Node1")),
        ]
    )
    def test_recursive_graph_iterator(self, _: str, reverse: bool, expected: tuple[str, ...]):
        iterator = traversal.RecursiveGraphIterator(self.graph)
        if reverse:
            iterator = reversed(iterator)
        nodes = list(iterator)
        self.assertEqual(tuple(node.op_type for node in nodes), expected)

    @parameterized.parameterized.expand(
        [
            ("forward", False, ("Node1", "Node2", "If")),
            ("reversed", True, ("If", "Node2", "Node1")),
        ]
    )
    def test_recursive_graph_iterator_recursive_controls_recursive_behavior(
        self, _: str, reverse: bool, expected: list[str]
    ):
        nodes = list(
            traversal.RecursiveGraphIterator(
                self.graph, recursive=lambda node: node.op_type != "If", reverse=reverse
            )
        )
        self.assertEqual(tuple(node.op_type for node in nodes), expected)


    def test_recursive_graph_iterator_with_multiple_graphs(self):
        graph_with_multiple_graphs = ir.Graph(
            [],
            [],
            nodes=[
                ir.Node(
                    "",
                    "Loop",
                    [],
                    attributes=[
                        ir.AttrGraphs(
                            "body_branches",
                            [
                                ir.Graph(
                                    [],
                                    [],
                                    nodes=[ir.Node("", "Node7", []), ir.Node("", "Node8", [])],
                                    name="body_graph_1",
                                ),
                                ir.Graph(
                                    [],
                                    [],
                                    nodes=[ir.Node("", "Node9", []), ir.Node("", "Node10", [])],
                                    name="body_graph_2",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            name="main_graph_with_multiple_graphs",
        )
        iterator = traversal.RecursiveGraphIterator(graph_with_multiple_graphs)
        nodes = list(iterator)
        expected = ("Loop", "Node7", "Node8", "Node9", "Node10")
        self.assertEqual(tuple(node.op_type for node in nodes), expected)


if __name__ == "__main__":
    unittest.main()
