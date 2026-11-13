# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx.checker
import onnx.parser
import onnx_ir as ir

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _cast_constant_of_shape


class CastConstantOfShapeTest(unittest.TestCase):
    def test_cast_after_constant_of_shape_is_fused(self):
        input_model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 10> (constant)
            }
            """
        )
        onnx.checker.check_model(input_model_proto, True)
        model = ir.serde.deserialize_model(input_model_proto)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 10)
        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    def test_cast_after_constant_of_shape_without_value_is_fused(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 10> (constant)
            }
            """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 10)
        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    def test_cast_to_string_after_constant_of_shape_is_not_fused(self):
        """Cast to string should not be fused into ConstantOfShape as ConstantOfShape does not support STRING output."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (string[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 8> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        # Verify the model serializes cleanly without SerdeError
        output_proto = ir.serde.serialize_model(model)
        self.assertIsNotNone(output_proto)

    def test_cast_to_string_after_constant_of_shape_without_value_is_not_fused(self):
        """Cast to string without value should not be fused into ConstantOfShape as ConstantOfShape does not support STRING output."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (string[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 8> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        # Verify the model serializes cleanly without SerdeError
        output_proto = ir.serde.serialize_model(model)
        self.assertIsNotNone(output_proto)

    def test_cast_after_constant_of_shape_with_ref_attr_is_not_fused(self):
        """Cast after ConstantOfShape with a reference value attribute should not be fused or replaced with zero."""
        model_text = """
            <ir_version: 10, opset_import: ["pkg": 1, "": 17]>
            main_graph (int64[2] shape) => (float[1, 4] output) {
                output = pkg.my_func <fill = float[1] {3.0}> (shape)
            }
            <domain: "pkg", opset_import: ["": 17]>
            my_func <fill> (shape) => (output) {
                constant = ConstantOfShape <value: tensor = @fill> (shape)
                output = Cast <to = 1> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        function = next(iter(model.functions.values()))
        self.assertEqual([node.op_type for node in function], ["ConstantOfShape", "Cast"])
        constant_node = function[0]
        self.assertIn("value", constant_node.attributes)
        self.assertTrue(constant_node.attributes["value"].is_ref())
        self.assertEqual(constant_node.attributes["value"].ref_attr_name, "fill")

    def test_cast_after_constant_of_shape_with_non_scalar_value_is_not_fused(self):
        """Cast after ConstantOfShape with a non-scalar value attribute should not be fused."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[2] {1., 2.}>(input_x)
                output = Cast <to = 1> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])

    def test_cast_after_constant_of_shape_without_value_numerical_correctness(self):
        """Verify numerical correctness of legitimate ConstantOfShape without value cast fusion."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 1> (constant)
            }
        """
        original = ir.from_onnx_text(model_text)
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        testing.assert_numerically_equal(
            original,
            model,
            (np.array([1, 4], dtype=np.int64),),
        )

    def test_cast_with_ref_attr_to_after_constant_of_shape_is_not_fused(self):
        """Cast with reference target dtype attribute should not be fused."""
        model_text = """
            <ir_version: 10, opset_import: ["pkg": 1, "": 17]>
            main_graph (int64[2] shape) => (float[1, 4] output) {
                output = pkg.my_func <target = 1> (shape)
            }
            <domain: "pkg", opset_import: ["": 17]>
            my_func <target> (shape) => (output) {
                constant = ConstantOfShape <value: tensor = float[1] {1.0}> (shape)
                output = Cast <to: int = @target> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        function = next(iter(model.functions.values()))
        self.assertEqual([node.op_type for node in function], ["ConstantOfShape", "Cast"])
        cast_node = function[1]
        self.assertIn("to", cast_node.attributes)
        self.assertTrue(cast_node.attributes["to"].is_ref())
        self.assertEqual(cast_node.attributes["to"].ref_attr_name, "target")

    def test_cast_with_ref_attr_to_after_constant_of_shape_without_value_is_not_fused(self):
        """Cast with reference target dtype without value should not be fused."""
        model_text = """
            <ir_version: 10, opset_import: ["pkg": 1, "": 17]>
            main_graph (int64[2] shape) => (float[1, 4] output) {
                output = pkg.my_func <target = 1> (shape)
            }
            <domain: "pkg", opset_import: ["": 17]>
            my_func <target> (shape) => (output) {
                constant = ConstantOfShape (shape)
                output = Cast <to: int = @target> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        function = next(iter(model.functions.values()))
        self.assertEqual([node.op_type for node in function], ["ConstantOfShape", "Cast"])
        cast_node = function[1]
        self.assertIn("to", cast_node.attributes)
        self.assertTrue(cast_node.attributes["to"].is_ref())
        self.assertEqual(cast_node.attributes["to"].ref_attr_name, "target")

    def test_cast_to_undefined_after_constant_of_shape_is_not_fused(self):
        """Cast to undefined data type should not be fused into ConstantOfShape."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 0> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])

    def test_cast_to_undefined_after_constant_of_shape_without_value_is_not_fused(self):
        """Cast to undefined data type without value should not be fused into ConstantOfShape."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 0> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])

    def test_cast_after_constant_of_shape_with_string_value_is_not_fused(self):
        """Cast after ConstantOfShape with a string value tensor should not be fused."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = string[1] {"hello"}>(input_x)
                output = Cast <to = 1> (constant)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])


if __name__ == "__main__":
    unittest.main()
