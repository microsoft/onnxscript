import unittest

import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import cast_constant_of_shape


class CastConstantOfShapeTest(unittest.TestCase):
    def test_cast_after_constant_of_shape_is_fused(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 10> (constant)
            }
            """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 10)

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
        count = cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 10)


if __name__ == "__main__":
    unittest.main()
