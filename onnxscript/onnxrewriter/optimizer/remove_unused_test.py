import unittest

import onnx

from onnxrewriter import optimizer


class RemoveUnusedTest(unittest.TestCase):
    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")

    def test_remove_unused_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        self.assertEqual(len(model.graph.initializer), 1)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.initializer), 0)

    def test_partially_used_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }
        """
        )
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[0].op_type, "Split")

    def test_remove_unused_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] z) {
                z, indices = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 1)

    def test_remove_unused_optional_outputs_dropout_in_function(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] z)
            {
                z = pkg.custom.afunction (x)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (x) => (z)
            {
                z, indices = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[0].node), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "MaxPool")
        self.assertEqual(len(model.functions[0].node[0].output), 2)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[0].node), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "MaxPool")
        self.assertEqual(len(model.functions[0].node[0].output), 1)

    def test_remove_used_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] y, float[1, 1, 5, 5] z) {
                y, z = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)

    def test_remove_multiple_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 1)

    def test_remove_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z, float[1, 3, 5, 5] mean) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 2)

    def test_avoid_remove_non_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x) => (float[1, 3, 5, 5] z, float[1, 3, 5, 5] InvStdDev) {
                scale = Constant <value_ints=[3]> ()
                B = Constant <value_ints=[3]> ()
                z, mean, InvStdDev = LayerNormalization(x, scale, B)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)
        optimizer.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(len(model.graph.node[2].output), 3)


if __name__ == "__main__":
    unittest.main()
