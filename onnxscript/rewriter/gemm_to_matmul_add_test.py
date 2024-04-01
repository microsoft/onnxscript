import unittest

import onnx.parser

from onnxscript._legacy_ir import irbuilder
from onnxscript.rewriter import gemm_to_matmul_add


class ReshapeGemmReshapeTest(unittest.TestCase):
    def test_reshape_gemm_reshape_replace_when_nd_inputs_are_broadcastable(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 4, 512, 512] input_x, float[4, 512, 64] input_y, float[4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )

        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 4)

    def test_reshape_gemm_reshape_replace_when_nd_inputs_are_broadcastable_in_nested_function(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.custom": 1]>
            agraph (float[1, 4, 512, 512] input_x, float[4, 512, 64] input_y, float[4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                output = afunction (input_x, input_y, input_z)
            }
            <domain: "pkg.custom", opset_import: [ "" : 17]>
            afunction (input_x, input_y, input_z) => (output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        # Hack to put value_info in since parser does not support this experimental naming format
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_x",
                onnx.TensorProto.FLOAT,
                [1, 4, 512, 512],
            )
        )
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_y", onnx.TensorProto.FLOAT, [4, 512, 64]
            )
        )
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_z", onnx.TensorProto.FLOAT, [1, 4, 512, 64]
            )
        )

        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.functions), 1)
        self.assertEqual(len(ir.functions[0].nodes), 4)
        self.assertEqual(ir.functions[0].nodes[2].op_type, "MatMul")
        self.assertEqual(ir.functions[0].nodes[3].op_type, "Add")

    def test_reshape_gemm_reshape_remain_when_input_last_dim_and_second_last_dim_not_matched(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 4, 512, 512] input_x, float[4, 256, 64] input_y, float[4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(len(ir.graph.nodes), 5)

    def test_reshape_gemm_reshape_remain_when_inputs_are_not_broadcastable(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 2, 512, 512] input_x, float[4, 512, 64] input_y, float[4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(len(ir.graph.nodes), 5)

    def test_reshape_gemm_reshape_replace_when_inputs_are_broadcastable_with_one_in_dims(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4, 512, 512] input_x, float[1, 4, 512, 64] input_y, float[1, 4, 512, 64] input_z) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 4)
        self.assertEqual(ir.graph.nodes[2].op_type, "MatMul")
        self.assertEqual(ir.graph.nodes[3].op_type, "Add")

    def test_reshape_gemm_reshape_replace_when_first_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4] input_x, float[2, 3, 4, 5] input_y, float[2, 3, 5] input_z) => (float[2, 3, 5] output)
            {
                shape_a = Constant<value: tensor = int64[2] {1, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[3] {2, 3, 5}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 4)
        self.assertEqual(ir.graph.nodes[2].op_type, "MatMul")
        self.assertEqual(ir.graph.nodes[3].op_type, "Add")

    def test_reshape_gemm_reshape_replace_when_first_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[8] input_x, float[2, 3, 4, 5] input_y, float[2, 3, 5] input_z) => (float[2, 3, 5] output)
            {
                shape_a = Constant<value: tensor = int64[2] {2, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[3] {2, 3, 5}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(len(ir.graph.nodes), 5)

    def test_reshape_gemm_reshape_replace_when_second_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 5, 4] input_x, float[4] input_y, float[2, 3, 5] input_z) => (float[2, 3, 5] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 5, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[3] {2, 3, 5}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 1)
        self.assertEqual(len(ir.graph.nodes), 4)
        self.assertEqual(ir.graph.nodes[2].op_type, "MatMul")
        self.assertEqual(ir.graph.nodes[3].op_type, "Add")

    def test_reshape_gemm_reshape_replace_when_second_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 5, 4] input_x, float[10] input_y, float[2, 3, 5] input_z) => (float[2, 3, 5] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 5, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[3] {2, 3, 5}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(len(ir.graph.nodes), 5)

    def test_reshape_gemm_reshape_remain_when_output_is_not_matmul_broadcasted(
        self,
    ):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3, 5, 4] input_x, float[5] input_y, float[2, 3, 5] input_z) => (float[2, 4, 6] output)
            {
                shape_a = Constant<value: tensor = int64[4] {2, 3, 5, 4}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[3] {2, 4, 6}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        ir = irbuilder.build_ir(model)
        count = gemm_to_matmul_add.rule.apply_to_model(ir)
        self.assertEqual(count, 0)
        self.assertEqual(len(ir.graph.nodes), 5)


if __name__ == "__main__":
    unittest.main()
