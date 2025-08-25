# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

from onnxscript.rewriter.repeated_patterns import (
    find_largest_repeated_pattern,
    node_type_frequency,
)


class TestGraphPatternRepeated(unittest.TestCase):
    def test_repeated_pattern_asimple(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a2"]),
                    oh.make_node("Add", ["a2", "de"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b2"]),
                    oh.make_node("Add", ["b2", "de"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(({("", "Add"): 4, ("", "Neg"): 2}, {4: 1, 2: 3}, 2, [("", "Neg")]), h)
        h = find_largest_repeated_pattern(onx)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2])

    def test_repeated_pattern_asimple_match(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a2"]),
                    oh.make_node("Add", ["a2", "de"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b2"]),
                    oh.make_node("Add", ["b2", "de"], ["Z"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a"])],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, ["a"])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="un"),
                    onh.from_array(np.array([2], dtype=np.float32), name="de"),
                ],
            ),
        )
        for i, n in enumerate(onx.graph.node):
            n.name = f"i{i}"
        onnx.checker.check_model(onx)
        _indices, pattern = find_largest_repeated_pattern(onx, replace=True, all_instances=True)
        self.assertEqual(["Add", "Neg", "Add"], [n.op_type for n in pattern[0].node])
        self.assertEqual(["X", "de", "un"], pattern[0].input)
        self.assertEqual(["a3"], pattern[0].output)
        self.assertEqual(["RepeatedPattern"], [n.op_type for n in pattern[1].node])
        self.assertEqual(["X", "de", "un"], pattern[1].input)
        self.assertEqual(["a3"], pattern[1].output)
        self.assertEqual(["X", "de", "un"], pattern[1].node[0].input)
        self.assertEqual(["a3"], pattern[1].node[0].output)
        self.assertEqual(["repeated", "repeated"], [n.op_type for n in onx.graph.node])
        self.assertEqual([["X", "de", "un"], ["a3", "de", "un"]], [n.input for n in onx.graph.node])
        self.assertEqual([["a3"], ["Z"]], [n.output for n in onx.graph.node])

    def test_repeated_pattern_order_equal(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a21"]),
                    oh.make_node("Abs", ["a1"], ["a22"]),
                    oh.make_node("Add", ["a21", "a22"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Neg", ["b1"], ["b21"]),
                    oh.make_node("Abs", ["b1"], ["b22"]),
                    oh.make_node("Add", ["b21", "b22"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(
            (
                {("", "Add"): 4, ("", "Neg"): 2, ("", "Abs"): 2},
                {4: 1, 2: 4},
                2,
                [("", "Neg"), ("", "Abs")],
            ),
            h,
        )
        h = find_largest_repeated_pattern(onx)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_repeated_pattern_order_unequal(self):
        onx = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["a1"]),
                    oh.make_node("Neg", ["a1"], ["a21"]),
                    oh.make_node("Abs", ["a1"], ["a22"]),
                    oh.make_node("Add", ["a21", "a22"], ["a3"]),
                    oh.make_node("Add", ["a3", "un"], ["b1"]),
                    oh.make_node("Abs", ["b1"], ["b22"]),
                    oh.make_node("Neg", ["b1"], ["b21"]),
                    oh.make_node("Add", ["b21", "b22"], ["Z"]),
                ],
                "test",
                [],
                [],
            ),
        )
        h = node_type_frequency(onx)
        self.assertEqual(
            (
                {("", "Add"): 4, ("", "Neg"): 2, ("", "Abs"): 2},
                {4: 1, 2: 4},
                2,
                [("", "Neg"), ("", "Abs")],
            ),
            h,
        )
        h = find_largest_repeated_pattern(onx)
        indices, _nodes = h
        self.assertEqual(indices, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
