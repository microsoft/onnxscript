# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the _constructors module."""

import onnx

import unittest

from onnxscript import ir
from onnxscript.ir._convenience import insert_nodes_in_value


def _create_model(model_text: str) -> ir.Model:
    model = onnx.parser.parse_model(model_text)
    return ir.serde.deserialize_model(model)


class ConvenienceTest(unittest.TestCase):
    def test_insert_nodes_in_value(self):
        # Main graph
        input = ir.Input("input")
        node_A = ir.node("op_A", [input])
        node_B = ir.node("op_B", node_A.outputs, outputs=[ir.Value(name="B")])
        node_C = ir.node("op_C", node_B.outputs)

        # New sequence to insert
        input_2 = ir.Input("input_2")
        node_M = ir.node("op_M", [input_2])
        node_N = ir.node("op_N", node_M.outputs)

        # Insert nodes in B
        insert_nodes_in_value(node_B.outputs[0], [node_M, node_N])
        self.assertEqual(len(node_B.outputs), 1)
        self.assertEqual(node_B.outputs[0].consumers()[0].op_type, "op_M")
        self.assertEqual(len(node_C.inputs), 1)
        self.assertEqual(node_C.inputs[0].producer().op_type, "op_N")
        self.assertEqual(node_C.inputs[0].name, "B")

    def test_insert_nodes_in_value_in_graph(self):
        ir_model = _create_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant<value_float=2.0>()
                a, b = SplitNode(x)
                z = MergeNode(a, b, two)
            }
        """
        )

        # Sequence to insert.
        # Note inputs = [i1, i2] and outputs = [b.outputs[1], c.outputs[0]].
        i1, i2 = ir.Input("i1"), ir.Input("i2")
        a = ir.node("op_1", [i1, i2])
        b = ir.node("op_2", [a.outputs[0], i1], num_outputs=2)
        c = ir.node("op_3", [i2, b.outputs[0]])

        # Insert nodes in SplitNode.outputs
        target_node = ir_model.graph[1]
        insert_nodes_in_value(target_node.outputs, [a, b, c])

        # Check target_node outputs have been renamed
        new_i1, new_i2 = target_node.outputs
        self.assertEqual(new_i1.name, "i1")
        self.assertEqual(new_i2.name, "i2")

        # Check i1 and i2 have new users
        self.assertEqual(tuple(node.op_type for node in new_i1.consumers()), ("op_1", "op_2"))
        self.assertEqual(tuple(node.op_type for node in new_i2.consumers()), ("op_1", "op_3"))

        # Check outputs have been correctly renamed as previous values
        self.assertEqual(b.outputs[1].name, "a")
        self.assertEqual(c.outputs[0].name, "b")

        # Check nodes have been inserted in the graph
        self.assertEqual(len(ir_model.graph), 6)

    def test_insert_nodes_in_input(self):
        ir_model = _create_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant<value_float=2.0>()
                z = Add(x, two)
            }
        """
        )

        # Sequence to insert.
        x = ir.Input("new_x")
        node = ir.node("Mul", [x, x])

        # Insert nodes in graph.inputs
        insert_nodes_in_value(ir_model.graph[1].inputs[0], [node])
        self.assertEqual(node.outputs[0].name, "x")

        # Check input has been renamed
        self.assertEqual(ir_model.graph.inputs[0].name, "new_x")

        # Finally, check new graph is valid
        proto = ir.to_proto(ir_model)
        onnx.checker.check_model(proto, full_check=True)

    def test_insert_nodes_in_output(self):
        ir_model = _create_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant<value_float=2.0>()
                z = Add(x, two)
            }
        """
        )

        # Sequence to insert.
        x = ir.Input("new_z")
        node = ir.node("Mul", [x, x])

        # Insert nodes in graph.inputs
        insert_nodes_in_value(ir_model.graph.outputs[0], [node])
        self.assertEqual(ir_model.graph[1].outputs[0].name, "new_z")

        # Check output name is preserved
        self.assertEqual(ir_model.graph.outputs[0].name, "z")

    def test_value_error_for_wrong_number_of_points(self):
        ir_model = _create_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant<value_float=2.0>()
                a, b = SplitNode(x)
                z = MergeNode(a, b, two)
            }
        """
        )
        node = ir.node("op_M", [ir.Input("new_x"), ir.Input("new_y")])
        with self.assertRaisesRegex(ValueError, "The number of values and inputs"):
            insert_nodes_in_value(ir_model.graph[0].outputs, [node])

        with self.assertRaisesRegex(ValueError, "The number of values and outputs"):
            insert_nodes_in_value(ir_model.graph[1].outputs, [node])


if __name__ == "__main__":
    unittest.main()
