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
        nodes = list(traversal.RecursiveGraphIterator(self.graph, reverse=reverse))
        self.assertEqual(tuple(node.op_type for node in nodes), expected)

    @parameterized.parameterized.expand(
        [
            ("forward", False, ["main_graph", "then_graph", "else_graph"]),
            ("reversed", True, ["main_graph", "else_graph", "then_graph"]),
        ]
    )
    def test_recursive_graph_iterator_enter_graph_handler(self, _: str, reverse: bool, expected: list[str]):
        scopes = []

        def enter_graph_handler(graph):
            scopes.append(graph.name)

        for __ in traversal.RecursiveGraphIterator(
                self.graph, enter_graph_handler=enter_graph_handler, reverse=reverse
            ):
            pass
        self.assertEqual(scopes, expected)

    @parameterized.parameterized.expand(
        [
            ("forward", False, ["then_graph", "else_graph", "main_graph",]),
            ("reversed", True, ["else_graph", "then_graph", "main_graph"]),
        ]
    )
    def test_recursive_graph_iterator_exit_graph_handler(self, _: str, reverse: bool, expected: list[str]):
        scopes = []

        def exit_graph_handler(graph):
            scopes.append(graph.name)

        for __ in traversal.RecursiveGraphIterator(
                self.graph, exit_graph_handler=exit_graph_handler, reverse=reverse
            ):
            pass
        self.assertEqual(scopes, expected)


if __name__ == "__main__":
    unittest.main()
