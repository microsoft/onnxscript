# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.checker
import onnx.parser
import onnx_ir as ir
import parameterized

from onnxscript.rewriter.rules.common import _cast_constant_of_shape


class CastConstantOfShapeTest(unittest.TestCase):
    def _apply_cast(
        self, opset: int, output_type: str, target_dtype: int, *, has_value: bool
    ) -> tuple[int, ir.Model]:
        value_attribute = " <value: tensor = float[1] {1.0}>" if has_value else ""
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 10, opset_import: [ "" : {opset}]>
            agraph (int64[2] input_x) => ({output_type}[1, 4] output)
            {{
                constant = ConstantOfShape{value_attribute} (input_x)
                output = Cast <to = {target_dtype}> (constant)
            }}
            """
        )
        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        return count, model

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

    @parameterized.parameterized.expand(
        [
            ("int_above_uint8_range", "int32", "{300}", 2, np.array([300], np.int32)),
            ("negative_int_to_uint8", "int32", "{-1}", 2, np.array([-1], np.int32)),
            ("int64_to_int8", "int64", "{200}", 3, np.array([200], np.int64)),
        ]
    )
    def test_fused_value_matches_cast_of_out_of_range_constant(
        self, _, value_type, value, to, value_array
    ):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (output)
            {{
                constant = ConstantOfShape <value: tensor = {value_type}[1] {value}>(input_x)
                output = Cast <to = {to}> (constant)
            }}
            """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        fused_value = model.graph[0].attributes["value"].value
        self.assertEqual(fused_value.dtype, to)
        expected = value_array.astype(ir.DataType(to).numpy())
        np.testing.assert_array_equal(fused_value.numpy(), expected)

    def test_bfloat16_cast_is_not_fused_before_constant_of_shape_support(self):
        count, model = self._apply_cast(
            17, "bfloat16", onnx.TensorProto.BFLOAT16, has_value=True
        )
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])

    def test_float8_cast_is_not_fused_before_constant_of_shape_support(self):
        count, model = self._apply_cast(
            19, "float8e4m3fn", onnx.TensorProto.FLOAT8E4M3FN, has_value=False
        )
        self.assertEqual(count, 0)
        self.assertEqual([node.op_type for node in model.graph], ["ConstantOfShape", "Cast"])

    def test_float8_cast_is_fused_when_constant_of_shape_supports_it(self):
        count, model = self._apply_cast(
            20, "float8e4m3fn", onnx.TensorProto.FLOAT8E4M3FN, has_value=False
        )
        self.assertEqual(count, 1)
        onnx.checker.check_model(ir.serde.serialize_model(model), True)


if __name__ == "__main__":
    unittest.main()
