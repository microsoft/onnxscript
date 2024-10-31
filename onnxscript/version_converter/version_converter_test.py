# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.checker
import onnx.parser
import onnx.shape_inference

from onnxscript import ir
from onnxscript.version_converter import convert_version


class VersionConverter18to19Test(unittest.TestCase):
    def test_version_convert_compatible(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[1, 4, 512, 64] input_y) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[3] {4, 512, 64}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        target_version = 19
        new_model = convert_version(model, target_version=target_version)
        nodes = new_model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 19)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 19)
        self.assertEqual(nodes[4].op_type, "MatMul")
        self.assertEqual(nodes[4].version, 18)

        out_model_proto = ir.serde.serialize_model(new_model)
        onnx.checker.check_model(out_model_proto, full_check=1)
        onnx.save(out_model_proto, "test1.onnx")


class VersionConverter19to20Test(unittest.TestCase):
    def test_version_convert_compatible(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x) => (float[4, 257, 64, 2] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512, 1}>()
                reshape_x = Reshape (input_x, shape_a)
                dft = DFT <axis = 2, onesided = 1> (reshape_x)
                shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                output = Reshape (dft, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        target_version = 20
        new_model = convert_version(model, target_version=target_version)
        nodes = new_model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 19)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 19)
        self.assertEqual(nodes[2].op_type, "DFT")
        self.assertEqual(nodes[2].version, 20)

        out_model_proto = ir.serde.serialize_model(new_model)
        #onnx.checker.check_model(out_model_proto, full_check=1)
        onnx.save(out_model_proto, "test2.onnx")

if __name__ == "__main__":
    unittest.main()
