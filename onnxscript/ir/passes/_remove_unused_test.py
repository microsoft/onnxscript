# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

from onnxscript import ir
from onnxscript.ir.passes._remove_unused import RemoveUnused


class RemoveUnusedTest(unittest.TestCase):
    def test_purge_empty(self):
        graph = ir.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
            opset_imports={"": 1},
        )
        remove_unused = RemoveUnused(graph)
        remove_unused.purge()
        self.assertEqual(tuple(graph), ())

    def test_purge_a_single_node(self):
        v0 = ir.Value(name="v0")
        node0 = ir.Node("", "Node0", inputs=(v0,), num_outputs=1)
        node1 = ir.Node("", "Node1", inputs=(v0,), num_outputs=1)
        node2 = ir.Node("", "Node2", inputs=(v0,), num_outputs=0)
        node3 = ir.Node("", "Node3", inputs=(), num_outputs=1)
        node4 = ir.Node("", "Node4", inputs=(None,), num_outputs=1)
        graph = ir.Graph(
            (v0,),
            (node0.outputs[0], node3.outputs[0], node4.outputs[0]),
            nodes=(node0, node1, node2, node3, node4),
            opset_imports={"": 1},
        )
        remove_unused = RemoveUnused(graph)
        remove_unused.purge()
        self.assertEqual(tuple(graph), (node0, node3, node4))

    def test_purge_a_tree(self):
        v0 = ir.Value(name="v0")
        node0 = ir.Node("", "Node0", inputs=(v0,), num_outputs=1)
        node1 = ir.Node("", "Node1", inputs=(node0.outputs[0],), num_outputs=1)
        node2 = ir.Node("", "Node2", inputs=(node0.outputs[0],), num_outputs=1)
        graph = ir.Graph(
            (v0,),
            (),
            nodes=(node0, node1, node2),
            opset_imports={"": 1},
        )
        remove_unused = RemoveUnused(graph)
        remove_unused.purge()
        self.assertEqual(tuple(graph), ())

    def test_purge_subgraph_partial(self):
        v0 = ir.Value(name="va")
        v1 = ir.Value(name="vb")
        v2 = ir.Value(name="vc")
        v3 = ir.Value(name="vd")
        node0 = ir.Node("", "a", inputs=(v0,), num_outputs=1)
        node1 = ir.Node("", "b", inputs=(v1,), num_outputs=1)
        node2 = ir.Node("", "c", inputs=(v2,), num_outputs=1)
        node3 = ir.Node("", "d", inputs=(v3,), num_outputs=1)
        node4 = ir.Node("", "sub", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1)
        node5 = ir.Node("", "add", inputs=(node2.outputs[0], node3.outputs[0]), num_outputs=1)
        node6 = ir.Node("", ">", inputs=(node0.outputs[0], node1.outputs[0]), num_outputs=1)
        then_graph = ir.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(node4.outputs[0],),
            nodes=(node4,),
            name="then_graph",
        )
        else_graph = ir.Graph(
            inputs=(node2.outputs[0], node3.outputs[0]),
            outputs=(),
            nodes=(node5,),
            name="else_graph",
        )

        node7 = ir.Node(
            "",
            "if",
            inputs=(node6.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraphs("subgraphs", [then_graph, else_graph]),
            ],
        )
        main_graph = ir.Graph(
            inputs=(v0, v1, v2, v3),
            outputs=(node7.outputs[0],),
            nodes=(node0, node1, node2, node3, node6, node7),
            name="main_graph",
            opset_imports={"": 1},
        )
        remove_unused = RemoveUnused(main_graph)
        remove_unused.purge()
        self.assertEqual(tuple(main_graph), (node0, node1, node2, node3, node6, node7))
        self.assertEqual(tuple(then_graph), (node4,))
        self.assertEqual(tuple(else_graph), ())

    def test_purge_subgraph_all(self):
        v0 = ir.Value(name="v0")
        node0 = ir.Node("", "c", inputs=(v0,), num_outputs=1)
        node1 = ir.Node("", "sub", inputs=(node0.outputs[0],), num_outputs=1)
        node2 = ir.Node("", ">", inputs=(v0,), num_outputs=1)
        then_graph = ir.Graph(
            inputs=(node0.outputs[0],),
            outputs=(node1.outputs[0],),
            nodes=(node1,),
            name="then_graph",
        )
        node4 = ir.Node(
            "",
            "if",
            inputs=(node2.outputs[0],),
            num_outputs=1,
            attributes=[
                ir.AttrGraph("then_graph", then_graph),
            ],
        )
        main_graph = ir.Graph(
            inputs=(v0,),
            outputs=(),
            nodes=(node0, node2, node4),
            name="main_graph",
        )
        remove_unused = RemoveUnused(main_graph)
        remove_unused.purge()
        self.assertEqual(tuple(main_graph), ())
        self.assertEqual(tuple(then_graph), ())


if __name__ == "__main__":
    unittest.main()
