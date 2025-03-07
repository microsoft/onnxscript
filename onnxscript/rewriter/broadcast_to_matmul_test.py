# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.parser
import onnx.shape_inference
import parameterized

from onnxscript import ir
from onnxscript.rewriter import broadcast_to_matmul


def _infer_shapes(model: ir.Model) -> ir.Model:
    """Run shape inference on the IR model."""
    # TODO: Update when shape inference is supported on the IR
    return ir.serde.deserialize_model(
        onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model))
    )


class TwoReshapesMatMulReshapeTest(unittest.TestCase):
    def test_reshape_matmul_reshape_replace_when_nd_inputs_are_broadcastable(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
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
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    @parameterized.parameterized.expand(
        [
            (
                "0d",
                [],
                [1, 1],
                [],
                [1, 1],
                [1, 1],
                [1, 1],
            ),
            (
                "x_1d",
                [4],
                [1, 4],
                [4, 2],
                [4, 2],
                [1, 2],
                [1, 2],
            ),
            (
                "y_1d",
                [1, 4],
                [1, 4],
                [2],
                [4, 2],
                [1, 2],
                [1, 2],
            ),
            (
                "both_1d",
                [2],
                [1, 2],
                [2],
                [2, 1],
                [],
                [],
            ),
        ]
    )
    def test_reshape_matmul_reshape_does_not_replace_when_output_sizes_do_not_match(
        self,
        _: str,
        input_x_shape: list[int],
        shape_a: list[int],
        input_y_shape: list[int],
        shape_b: list[int],
        output_shape: list[int],
        shape_c: list[int],
    ):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float{input_x_shape} input_x, float{input_y_shape} input_y) => (float{output_shape} output)
            {{
                shape_a = Constant<value: tensor = int64[{len(shape_a)}] {{ {", ".join(str(i) for i in shape_a)} }}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[{len(shape_b)}] {{ {", ".join(str(i) for i in shape_b)} }}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[{len(shape_c)}] {{ {", ".join(str(i) for i in shape_c)} }}>()
                output = Reshape (matmul, shape_c)
            }}
            """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 7)
        model = _infer_shapes(model)
        self.assertEqual(model.graph.outputs[0].shape, output_shape)

    def test_reshape_matmul_reshape_replace_when_nd_inputs_are_broadcastable_in_nested_function(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[1, 4, 512, 512] input_x, float[1, 4, 512, 64] input_y) => (float[1, 4, 512, 64] output)
            {
                output = pkg.custom.afunction (input_x, input_y)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (input_x, input_y) => (output)
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
        # Hack to put value_info in since parser does not support this experimental naming format
        model_proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_x",
                onnx.TensorProto.FLOAT,
                [1, 4, 512, 512],
            )
        )
        model_proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_y", onnx.TensorProto.FLOAT, [1, 4, 512, 64]
            )
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[("pkg.custom", "afunction", "")]), 4)
        self.assertEqual(
            model.functions[("pkg.custom", "afunction", "")][-1].op_type, "MatMul"
        )

    def test_reshape_matmul_reshape_remain_when_input_last_dim_and_second_last_dim_not_matched(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 512, 4] input_x, float[4, 64, 512] input_y) => (float[1, 4, 512, 64] output)
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
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 7)

    def test_reshape_matmul_reshape_remain_one_reshape_when_inputs_are_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 8, 512, 64] input_x, float[4, 4, 64, 512] input_y) => (float[2, 8, 512, 512] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 8, 512, 64}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[4] {2, 8, 64, 512}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {2, 8, 512, 512}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        # subset pattern matched
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_matmul_reshape_replace_when_inputs_are_broadcastable_with_one_in_dims(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 8, 512, 64] input_x, float[1, 1, 2, 8, 64, 512] input_y) => (float[1, 1, 2, 8, 512, 512] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 8, 512, 64}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[6] {1, 1, 2, 8, 64, 512}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[6] {1, 1, 2, 8, 512, 512}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_matmul_reshape_replace_when_first_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4] input_x, float[2, 3, 4, 5] input_y) => (float[2, 3, 5] output)
            {
                shape_a = Constant<value: tensor = int64[2] {1, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[4] {2, 3, 4, 5}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[3] {2, 3, 5}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_matmul_reshape_replace_when_first_input_is_one_dimension_and_second_isexpanded_alike_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[5] input_x, float[5, 1] input_y) => (float[1] output)
            {
                shape_a = Constant<value: tensor = int64[2] {1, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[2] {5, 1}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[1] {1}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_matmul_reshape_remain_when_first_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[8] input_x, float[2, 3, 4, 5] input_y) => (float[2, 3, 2, 5] output)
            {
                shape_a = Constant<value: tensor = int64[2] {2, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[4] {2, 3, 4, 5}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {2, 3, 2, 5}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 7)

    def test_reshape_matmul_reshape_replace_when_second_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 4, 5] input_x, float[5] input_y) => (float[2, 3, 4] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 4, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[2] {5, 1}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[3] {2, 3, 4}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_matmul_reshape_remain_one_reshape_when_second_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 4, 5] input_x, float[10] input_y) => (float[2, 3, 4, 2] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 4, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[2] {5, 2}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {2, 3, 4, 2}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        # subset pattern matched
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_matmul_reshape_remain_when_output_is_not_matmul_broadcasted(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 4, 5] input_x, float[5, 8] input_y) => (float[2, 4, 6, 4] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 4, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[2] {5, 2}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {2, 4, 6, 4}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 7)


class OneReshapeMatMulReshapeTest(unittest.TestCase):
    def test_reshape_matmul_reshape_replace_when_nd_inputs_are_broadcastable(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 512, 4096] input_x, float[4096, 4096] input_y) => (float[1, 512, 4096] output)
            {
                shape_a = Constant<value: tensor = int64[3] {1, 512, 4096}>()
                reshape_x = Reshape (input_x, shape_a)
                matmul = MatMul (reshape_x, input_y)
                shape_c = Constant<value: tensor = int64[3] {1, 512, 4096}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = broadcast_to_matmul.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        # The constant nodes are not removed. They should be removed by a subsequent DCE in optimizer.
        self.assertEqual(len(model.graph), 3)


if __name__ == "__main__":
    unittest.main()
