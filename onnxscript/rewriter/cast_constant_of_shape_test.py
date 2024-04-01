import unittest

import onnx.parser

from onnxscript._legacy_ir import irbuilder
from onnxscript.rewriter import cast_constant_of_shape


class CastConstantOfShapeTest(unittest.TestCase):
    def test_cast_after_constant_of_shape_is_fused(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 10> (constant)
            }
            """
        )
        ir = irbuilder.build_ir(model)
        count = cast_constant_of_shape.rules.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 1)
        self.assertEqual(ir.graph.nodes[0].attributes["value"].data_type, 10)

    def test_cast_after_constant_of_shape_without_value_is_fused(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float16[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 10> (constant)
            }
            """
        )
        ir = irbuilder.build_ir(model)
        count = cast_constant_of_shape.rules.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 1)
        self.assertEqual(ir.graph.nodes[0].attributes["value"].data_type, 10)


if __name__ == "__main__":
    unittest.main()
