# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx.checker
import onnx.parser
import onnx_ir as ir

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
        """Cast to string should not be fused into ConstantOfShape as string serialization fails."""
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
        """Cast to string without value should not be fused into ConstantOfShape."""
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


if __name__ == "__main__":
    unittest.main()
