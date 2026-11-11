# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.checker
import onnx.parser
import onnx_ir as ir
import parameterized

from onnxscript import rewriter
from onnxscript.rewriter import pattern
from onnxscript.rewriter.rules.common import _cast_constant_of_shape


class CastConstantOfShapeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("target_without_fill", False, True),
            ("target_with_fill", True, True),
            ("fill", True, False),
        ]
    )
    def test_function_reference_attributes_are_preserved(self, _, explicit_fill, ref_target):
        shape = ir.val("shape", ir.DataType.INT64, [1])
        fill = ir.AttrTensor("value", ir.tensor(np.array([2.0], dtype=np.float32)))
        if not ref_target:
            fill = ir.RefAttr("value", "parameter", ir.AttributeType.TENSOR)
        constant = ir.Node("", "ConstantOfShape", [shape], [fill] if explicit_fill else [])
        dtype = (
            ir.RefAttr("to", "parameter", ir.AttributeType.INT)
            if ref_target
            else ir.AttrInt64("to", ir.DataType.FLOAT16)
        )
        cast = ir.Node("", "Cast", constant.outputs, [dtype])
        function = ir.Function(
            "local",
            "F",
            graph=ir.Graph(
                [shape], cast.outputs, nodes=[constant, cast], opset_imports={"": 18}
            ),
            attributes=[
                ir.Attr(
                    "parameter",
                    ir.AttributeType.INT if ref_target else ir.AttributeType.TENSOR,
                    None,
                )
            ],
        )
        model = ir.Model(
            ir.Graph([], [], nodes=[], name="main", opset_imports={"": 18, "local": 1}),
            ir_version=10,
            functions=[function],
        )
        onnx.checker.check_model(ir.to_proto(model))
        before = ir.to_proto(model).SerializeToString()
        self.assertEqual(_cast_constant_of_shape.rules.apply_to_model(model), 0)
        rewriter.RewritePass(rewriter._DEFAULT_REWRITE_RULES)(model)
        self.assertEqual(ir.to_proto(model).SerializeToString(), before)

        # Replacement functions also reject references when used without a condition.
        original_rule = (
            _cast_constant_of_shape.cast_constant_of_shape_rule
            if explicit_fill
            else _cast_constant_of_shape.cast_constant_of_shape_without_value_rule
        )
        unguarded_rule = pattern.RewriteRule(
            original_rule._target_pattern, original_rule._replacement_pattern
        )
        self.assertIsNone(unguarded_rule.try_rewrite(model, function, cast))

        if ref_target:
            cast.attributes["to"] = ir.AttrInt64("to", ir.DataType.FLOAT16)
        else:
            constant.attributes["value"] = ir.AttrTensor(
                "value", ir.tensor(np.array([2.0], dtype=np.float32))
            )
        self.assertEqual(_cast_constant_of_shape.rules.apply_to_model(model), 1)
        onnx.checker.check_model(ir.to_proto(model))

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


if __name__ == "__main__":
    unittest.main()
