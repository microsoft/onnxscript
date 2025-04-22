# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx
import parameterized

import onnxscript.optimizer
from onnxscript import ir


@parameterized.parameterized_class(("using_ir",), [(False,), (True,)])
class RemoveUnusedTest(unittest.TestCase):
    using_ir: bool

    def remove_unused_nodes(
        self, model: onnx.ModelProto, remove_initialized_inputs: bool = False
    ):
        if self.using_ir:
            model_ir = ir.serde.deserialize_model(model)
            onnxscript.optimizer.remove_unused_nodes(model_ir, remove_initialized_inputs)
            model = ir.serde.serialize_model(model_ir)
            return model
        onnxscript.optimizer.remove_unused_nodes(model, remove_initialized_inputs)
        return model

    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")

    def test_remove_unused_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        self.assertEqual(len(model.graph.initializer), 1)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.initializer), 0)

    def test_unused_initialized_inputs_are_removed_when_requested(self):
        # https://github.com/microsoft/onnxscript/issues/2211
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] two) => (float[N] z)
            <float two = {2.0,2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model, remove_initialized_inputs=True)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.input), 1)

    def test_unused_initialized_inputs_are_kept_by_default(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] two) => (float[N] z)
            <float two = {2.0,2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.input), 2)

    @parameterized.parameterized.expand([True, False])
    def test_unused_inputs_are_not_removed(self, remove_initialized_inputs: bool):
        # preserve inputs as part of interface
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x, float[N] two) => (float[N] z)
            {
                four = Add(two, two)
                z = Mul(x, x)
            }
        """
        )
        model = self.remove_unused_nodes(
            model, remove_initialized_inputs=remove_initialized_inputs
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "Mul")
        self.assertEqual(len(model.graph.input), 2)

    def test_partially_used_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }
        """
        )
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[0].op_type, "Split")

    def test_remove_unused_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] z) {
                z, indices = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(model.graph.node[0].output, ["z"])

    def test_remove_unused_optional_outputs_dropout_in_function(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17, "pkg.custom": 1]>
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
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[0].node), 1)
        self.assertEqual(model.functions[0].node[0].op_type, "MaxPool")
        self.assertEqual(model.functions[0].node[0].output, ["z"])

    def test_remove_used_optional_outputs_maxpool(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 1, 5, 5] x) => (float[1, 1, 5, 5] y, float[1, 1, 5, 5] z) {
                y, z = MaxPool <pads = [2, 2, 2, 2], kernel_shape = [5, 5]> (x)
            }
        """
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(len(model.graph.node[0].output), 2)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "MaxPool")
        self.assertEqual(model.graph.node[0].output, ["y", "z"])

    def test_remove_multiple_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
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
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z"])

    def test_remove_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
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
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z", "mean"])

    def test_avoid_remove_non_trailing_unused_optional_outputs_layernorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
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
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 3)
        self.assertEqual(model.graph.node[2].op_type, "LayerNormalization")
        self.assertEqual(list(model.graph.node[2].output), ["z", "", "InvStdDev"])

    def test_remove_trailing_unused_optional_outputs_batchnorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x, float[3] scale, float[3] B) => (float[1, 3, 5, 5] z) {
                z, mean_out, var_out = BatchNormalization <training_mode=1> (x, scale, B, mean, var)
            }
        """
        )
        self.assertEqual(len(model.graph.node[0].attribute), 1)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "BatchNormalization")
        # Check that both the mean/var outputs are removed, and training_mode attribute is removed.
        self.assertEqual(list(model.graph.node[0].output), ["z"])
        self.assertEqual(len(model.graph.node[0].attribute), 0)

    def test_avoid_remove_used_optional_outputs_batchnorm(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 17]>
            agraph (float[1, 3, 5, 5] x, float[3] scale, float[3] B) => (float[1, 3, 5, 5] z, float[3] mean_out, float[3] var_out) {
                z, mean_out, var_out = BatchNormalization <training_mode=1> (x, scale, B, mean, var)
            }
        """
        )
        self.assertEqual(len(model.graph.node[0].attribute), 1)
        model = self.remove_unused_nodes(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "BatchNormalization")
        # Check that the mean/var outputs are NOT removed, and training_mode attribute is NOT removed.
        self.assertEqual(list(model.graph.node[0].output), ["z", "mean_out", "var_out"])
        self.assertEqual(len(model.graph.node[0].attribute), 1)


if __name__ == "__main__":
    unittest.main()
