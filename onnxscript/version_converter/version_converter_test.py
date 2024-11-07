# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.checker
import onnx.parser
import onnx.shape_inference

from onnxscript import ir, version_converter


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
        version_converter.convert_version(model, target_version=target_version)
        nodes = model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 19)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 19)
        self.assertEqual(nodes[4].op_type, "MatMul")
        self.assertEqual(nodes[4].version, 19)


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
        version_converter.convert_version(model, target_version=target_version)
        nodes = model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 20)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 20)
        self.assertEqual(nodes[2].op_type, "DFT")
        self.assertEqual(nodes[2].version, 20)

    def test_version_convert_gridsample_linear(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 1024, 1024] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[5] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bilinear"> (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {4, 1024, 1024}>()
                output = Reshape (gridsample, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(model.graph._nodes[4].op_type, "GridSample")
        self.assertEqual(model.graph._nodes[4]._attributes["mode"].value, "bilinear")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        nodes = model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 20)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 20)
        self.assertEqual(nodes[4].op_type, "GridSample")
        self.assertEqual(nodes[4].version, 20)
        self.assertEqual(model.graph._nodes[4]._attributes["mode"].value, "linear")

    def test_version_convert_gridsample_cubic(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 1024, 1024] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[5] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bicubic"> (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {4, 1024, 1024}>()
                output = Reshape (gridsample, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        self.assertEqual(model.graph._nodes[4].op_type, "GridSample")
        self.assertEqual(model.graph._nodes[4]._attributes["mode"].value, "bicubic")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        nodes = model.graph._nodes

        self.assertEqual(nodes[0].op_type, "Constant")
        self.assertEqual(nodes[0].version, 20)
        self.assertEqual(nodes[1].op_type, "Reshape")
        self.assertEqual(nodes[1].version, 20)
        self.assertEqual(nodes[4].op_type, "GridSample")
        self.assertEqual(nodes[4].version, 20)
        self.assertEqual(model.graph._nodes[4]._attributes["mode"].value, "cubic")


if __name__ == "__main__":
    unittest.main()
