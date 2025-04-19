# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.defs
import onnx.parser

from onnxscript import ir, version_converter


class AdapterCoverageTest(unittest.TestCase):
    def get_all_unique_schema_versions(self) -> dict[str, list]:
        """Collect all unique versions of ONNX standard domain ops"""
        op_version_dict = {}
        all_schemas = onnx.defs.get_all_schemas_with_history()
        for schema in all_schemas:
            if schema.name not in op_version_dict:
                op_version_dict[schema.name] = [schema.since_version]
            else:
                if schema.since_version not in op_version_dict[schema.name]:
                    op_version_dict[schema.name].append(schema.since_version)
        return op_version_dict

    # TODO(shubhambhokare1) : Using existing onnx testing suite to verify operator adapter's functionality
    def test_upstream_coverage(self):
        op_version_dict = self.get_all_unique_schema_versions()
        op_upgrades = []
        for op_type in op_version_dict:  # pylint: disable=consider-using-dict-items
            for opset_version in op_version_dict[op_type]:
                op_upgrades.append((op_type, opset_version))

        adapter_list = version_converter._version_converter.registry.op_adapters  # pylint: disable=protected-access
        for adapter_sig in adapter_list:
            adapter_info = list(adapter_sig)
            domain, name, upgrade_version = (
                adapter_info[0],
                adapter_info[1],
                adapter_info[2] + 1,
            )
            self.assertEqual(domain, "")
            self.assertIn((name, upgrade_version), op_upgrades)

    def test_version_convert_non_standard_onnx_domain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "local" : 1]>
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
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bilinear")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, None)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, None)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, None)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bilinear")


class VersionConverter18to17Test(unittest.TestCase):
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
        target_version = 17
        version_converter.convert_version(model, target_version=target_version)


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

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 19)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 19)
        self.assertEqual(model.graph.node(4).op_type, "MatMul")
        self.assertEqual(model.graph.node(4).version, 19)


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

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(2).op_type, "Constant")
        self.assertEqual(model.graph.node(3).version, 20)
        self.assertEqual(model.graph.node(3).op_type, "DFT")
        self.assertEqual(model.graph.node(3).version, 20)
        self.assertEqual(len(model.graph.node(3).inputs), 2)

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
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bilinear")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, 20)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "linear")

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
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bicubic")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, 20)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "cubic")

    def test_version_convert_inline(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 257, 64, 2] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[5] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bilinear"> (reshape_x, reshape_y)
                output = foo(gridsample)
            }

            <opset_import: [ "" : 18]>
            foo (x) => (dft) {
                dft = DFT <axis = 2, onesided = 1> (x)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        target_version = 20
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, 20)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "linear")
        self.assertEqual(model.graph.node(6).op_type, "DFT")
        self.assertEqual(model.graph.node(6).version, 20)
        self.assertEqual(len(model.graph.node(6).inputs), 2)


class VersionConverter20to21Test(unittest.TestCase):
    def test_version_groupnorm(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[2] scale, float[2] bias) => (float[4, 512, 512] output)
            {
                groupnorm = GroupNormalization <num_groups = 2> (input_x, scale, bias)
                shape_c = Constant<value: tensor = int64[4] {4, 512, 512}>()
                output = Reshape (groupnorm, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        target_version = 21
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(3).op_type, "Reshape")
        self.assertEqual(model.graph.node(3).version, 21)
        self.assertEqual(model.graph.node(4).op_type, "Expand")
        self.assertEqual(model.graph.node(4).version, 21)
        self.assertEqual(model.graph.node(5).op_type, "Reshape")
        self.assertEqual(model.graph.node(5).version, 21)
        self.assertEqual(model.graph.node(6).op_type, "Reshape")
        self.assertEqual(model.graph.node(6).version, 21)
        self.assertEqual(model.graph.node(7).op_type, "Expand")
        self.assertEqual(model.graph.node(7).version, 21)
        self.assertEqual(model.graph.node(8).op_type, "Reshape")
        self.assertEqual(model.graph.node(8).version, 21)
        self.assertEqual(model.graph.node(9).op_type, "GroupNormalization")
        self.assertEqual(model.graph.node(9).version, 21)

    def test_version_groupnorm_no_bias(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[2] scale) => (float[4, 512, 512] output)
            {
                groupnorm = GroupNormalization <num_groups = 2> (input_x, scale)
                shape_c = Constant<value: tensor = int64[4] {4, 512, 512}>()
                output = Reshape (groupnorm, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        target_version = 21
        version_converter.convert_version(model, target_version=target_version)

        self.assertEqual(model.graph.node(0).op_type, "GroupNormalization")
        self.assertEqual(model.graph.node(0).version, 20)


class VersionConverter23to24Test(unittest.TestCase):
    def test_version_convert_compatible(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 23]>
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
        target_version = 24
        version_converter.convert_version(model, target_version=target_version)


if __name__ == "__main__":
    unittest.main()
