# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.reference
import onnx_ir as ir

import onnxscript.optimizer as optimizer


class OptimizerTest(unittest.TestCase):
    def test_preserves_magnitude_stabilizer(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: ["" : 18]>
            magnitude (float[3] real, float[3] imag) => (float[3] output)
            {
                exponent = Constant<value = float {2.0}>()
                epsilon = Constant<value = float {1e-12}>()
                real_squared = Pow(real, exponent)
                imag_squared = Pow(imag, exponent)
                sum_squared = Add(real_squared, imag_squared)
                stabilized = Add(sum_squared, epsilon)
                output = Sqrt(stabilized)
            }
            """
        )
        inputs = {
            "real": np.array([0.0, 1e-8, 1.0], dtype=np.float32),
            "imag": np.array([0.0, 1e-8, 1.0], dtype=np.float32),
        }
        original = ir.serde.serialize_model(model)
        expected = onnx.reference.ReferenceEvaluator(original).run(None, inputs)[0]

        optimizer.optimize_ir(model)
        optimized = ir.serde.serialize_model(model)
        actual = onnx.reference.ReferenceEvaluator(optimized).run(None, inputs)[0]

        self.assertGreater(expected[0], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_folded_from_key_is_accessible(self):
        """Test that FOLDED_FROM_KEY is accessible from the public API."""
        self.assertTrue(hasattr(optimizer, "FOLDED_FROM_KEY"))
        self.assertIsInstance(optimizer.FOLDED_FROM_KEY, str)
        self.assertEqual(optimizer.FOLDED_FROM_KEY, "pkg.onnxscript.optimizer.folded_from")

    def _model_proto(self) -> onnx.ModelProto:
        return onnx.parser.parse_model(
            """
                <
                    ir_version: 8,
                    opset_import: ["pkg.onnxscript.torch_lib" : 1, "" : 18, "pkg.onnxscript.torch_lib.common" : 1],
                    producer_name: "pytorch",
                    producer_version: "2.2.0"
                >
                main_graph (float[3,5] l_tensor_x_) => (float[3,5] return_val)
                < _val_2, float[3,5] l_tensor_x_, float[2,5] getitem, float[1,5] getitem_1>
                {
                    _val_1 = Constant <value: tensor = int64 {2}> ()
                    _val_2 = pkg.onnxscript.torch_lib.aten_split <dim: int = 0> (l_tensor_x_, _val_1)
                    _val_3 = Constant <value: tensor = int64 {0}> ()
                    getitem = pkg.onnxscript.torch_lib.aten_getitem (_val_2, _val_3)
                    _val_5 = Constant <value: tensor = int64 {1}> ()
                    getitem_1 = pkg.onnxscript.torch_lib.aten_getitem (_val_2, _val_5)
                    return_val = Concat <axis: int = 0> (getitem_1, getitem)
                }

                <domain: "pkg.onnxscript.torch_lib", opset_import: ["" : 18]>
                aten_split (self, split_size) => (return_val)
                {
                    return_val = SplitToSequence <axis: int = @dim> (self, split_size)
                }

                <domain: "pkg.onnxscript.torch_lib", opset_import: ["" : 18]>
                aten_getitem (self, i) => (return_val)
                {
                    return_val = SequenceAt (self, i)
                }

                <domain: "pkg.onnxscript.torch_lib.common", opset_import: ["" : 18]>
                Rank (input) => (return_val)
                {
                    tmp = Shape (input)
                    return_val = Size (tmp)
                }

                <domain: "pkg.onnxscript.torch_lib.common", opset_import: ["" : 18]>
                IsScalar (input) => (return_val)
                {
                    tmp = Shape (input)
                    tmp_0 = Size (tmp)
                    tmp_1 = Constant <value_int: int = 0> ()
                    return_val = Equal (tmp_0, tmp_1)
                }
                """
        )

    def test_static_split_to_sequence_with_uneven_split_proto(self):
        model_proto = self._model_proto()
        optimized = optimizer.optimize(
            model_proto, num_iterations=1, onnx_shape_inference=False
        )
        self.assertEqual(len(optimized.graph.node), 2)
        self.assertEqual(len(optimized.graph.node[0].output), 2)
        self.assertEqual(optimized.graph.node[0].op_type, "Split")

    def test_static_split_to_sequence_with_uneven_split_ir(self):
        model_proto = self._model_proto()
        model_ir = ir.serde.deserialize_model(model_proto)
        optimizer.optimize_ir(model_ir, num_iterations=1, onnx_shape_inference=False)
        self.assertEqual(len(model_ir.graph), 2)
        self.assertEqual(len(model_ir.graph.node(0).outputs), 2)
        self.assertEqual(model_ir.graph.node(0).op_type, "Split")


if __name__ == "__main__":
    unittest.main()
