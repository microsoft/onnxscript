# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
            ("reversed", True, ("If", "Node6", "Node5", "Node4", "Node3", "Node2", "Node1")),
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


if __name__ == "__main__":
    unittest.main()
