# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.parser
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from onnxscript.rewriter.rules.common import _gemm_to_matmul_add


def _run_model(proto: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    outputs = ReferenceEvaluator(proto).run(None, feeds)
    assert isinstance(outputs, list)
    return outputs


class ReshapeGemmReshapeTest(unittest.TestCase):
    def test_reshape_gemm_reshape_remain_when_transpose_and_final_shape_is_inferred(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 2] X, float[2, 2] Y, float[2, 2] C) => (float[2, 2] O)
            {
                sa = Constant<value: tensor = int64[2] {2, 2}>()
                A = Reshape(X, sa)
                M = Gemm<alpha=1.0, beta=1.0, transB=1>(A, Y, C)
                sc = Constant<value: tensor = int64[2] {2, -1}>()
                O = Reshape(M, sc)
            }
        """
        )
        feeds = {
            "X": np.eye(2, dtype=np.float32),
            "Y": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "C": np.zeros((2, 2), dtype=np.float32),
        }
        expected = _run_model(model_proto, feeds)[0]

        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)

        self.assertEqual(count, 0)
        actual = _run_model(ir.serde.serialize_model(model), feeds)[0]
        np.testing.assert_array_equal(expected, np.array([[1, 3], [2, 4]], dtype=np.float32))
        np.testing.assert_array_equal(actual, expected)

    def test_reshape_gemm_reshape_replace_when_nd_inputs_are_broadcastable(self):
        model_proto = onnx.parser.parse_model(
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

        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_gemm_reshape_replace_when_nd_inputs_are_broadcastable_in_nested_function(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model_proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_x",
                onnx.TensorProto.FLOAT,
                [1, 4, 512, 512],
            )
        )
        model_proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_y", onnx.TensorProto.FLOAT, [4, 512, 64]
            )
        )
        model_proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "pkg.custom::afunction/input_z", onnx.TensorProto.FLOAT, [1, 4, 512, 64]
            )
        )

        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(len(model.functions[("pkg.custom", "afunction", "")]), 4)
        self.assertEqual(model.functions[("pkg.custom", "afunction", "")][2].op_type, "MatMul")
        self.assertEqual(model.functions[("pkg.custom", "afunction", "")][3].op_type, "Add")

    def test_reshape_gemm_reshape_remain_when_input_last_dim_and_second_last_dim_not_matched(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_remain_when_inputs_are_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_replace_when_inputs_are_broadcastable_with_one_in_dims(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)
        self.assertEqual(model.graph[2].op_type, "MatMul")
        self.assertEqual(model.graph[3].op_type, "Add")

    def test_reshape_gemm_reshape_replace_when_first_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)
        self.assertEqual(model.graph[2].op_type, "MatMul")
        self.assertEqual(model.graph[3].op_type, "Add")

    def test_reshape_gemm_reshape_remain_when_first_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_replace_when_second_input_is_one_dimension_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)
        self.assertEqual(model.graph[2].op_type, "MatMul")
        self.assertEqual(model.graph[3].op_type, "Add")

    def test_reshape_gemm_reshape_remain_when_second_input_is_one_dimension_and_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_replaces_when_inputs_are_two_dimensional_and_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3, 5] input_x, float[5, 10] input_y, float[3, 10] input_z) => (float[3, 10] output)
            {
                shape_a = Constant<value: tensor = int64[2] {3, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[2] {3, 10}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        replacement_count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(replacement_count, 1)
        self.assertEqual(len(model.graph), 4)

    def test_reshape_gemm_reshape_remain_when_inputs_are_two_dimension_and_not_broadcastable(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[5, 3] input_x, float[5, 10] input_y, float[3, 10] input_z) => (float[3, 10] output)
            {
                shape_a = Constant<value: tensor = int64[2] {3, 5}>()
                reshape_x = Reshape (input_x, shape_a)
                gemm = Gemm<alpha=1.0, beta=1.0> (reshape_x, input_y, input_z)
                shape_d = Constant<value: tensor = int64[2] {3, 10}>()
                output = Reshape (gemm, shape_d)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_remain_when_output_is_not_matmul_broadcasted(
        self,
    ):
        model_proto = onnx.parser.parse_model(
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
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_remain_when_bias_broadcasting_expands_output_rank(self):
        # input_x: [2], input_y: [2, 3], bias C: [1, 3] with final reshape to [3].
        # MatMul(input_x, input_y) produces [3], but adding C broadcasts to [1, 3].
        # Removing the final Reshape would expand the output rank from (3,) to (1, 3).
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2] X, float[2, 3] Y, float[1, 3] C) => (float[3] O)
            {
                sa = Constant<value: tensor = int64[2] {1, 2}>()
                A = Reshape(X, sa)
                M = Gemm<alpha=1.0, beta=1.0>(A, Y, C)
                sc = Constant<value: tensor = int64[1] {-1}>()
                O = Reshape(M, sc)
            }
        """
        )
        feeds = {
            "X": np.array([1, 2], dtype=np.float32),
            "Y": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "C": np.array([[10, 20, 30]], dtype=np.float32),
        }
        expected = _run_model(model_proto, feeds)[0]
        self.assertEqual(expected.shape, (3,))

        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

        actual = _run_model(ir.serde.serialize_model(model), feeds)[0]
        np.testing.assert_array_equal(expected, actual)

    def test_reshape_gemm_reshape_replace_when_bias_vector_matches_final_shape(self):
        # Control case: vector bias C has shape [3], matching expected final output shape [3].
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2] X, float[2, 3] Y, float[3] C) => (float[3] O)
            {
                sa = Constant<value: tensor = int64[2] {1, 2}>()
                A = Reshape(X, sa)
                M = Gemm<alpha=1.0, beta=1.0>(A, Y, C)
                sc = Constant<value: tensor = int64[1] {-1}>()
                O = Reshape(M, sc)
            }
        """
        )
        feeds = {
            "X": np.array([1, 2], dtype=np.float32),
            "Y": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "C": np.array([10, 20, 30], dtype=np.float32),
        }
        expected = _run_model(model_proto, feeds)[0]
        self.assertEqual(expected.shape, (3,))

        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 4)

        actual = _run_model(ir.serde.serialize_model(model), feeds)[0]
        np.testing.assert_allclose(expected, actual)

    def test_reshape_gemm_reshape_remain_when_bias_shape_has_symbolic_dim(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2] X, float[2, 3] Y, float[N] C) => (float[3] O)
            {
                sa = Constant<value: tensor = int64[2] {1, 2}>()
                A = Reshape(X, sa)
                M = Gemm<alpha=1.0, beta=1.0>(A, Y, C)
                sc = Constant<value: tensor = int64[1] {3}>()
                O = Reshape(M, sc)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 5)

    def test_reshape_gemm_reshape_remain_when_local_function_uses_ref_attr_trans(self):
        model_text = """
        <ir_version: 7, opset_import: [ "" : 17, "custom" : 1]>
        agraph (float[2] X, float[2, 3] Y, float[3] C) => (float[3] O) {
            O = custom.my_gemm_func (X, Y, C)
        }
        <domain: "custom", opset_import: [ "" : 17]>
        my_gemm_func <transB: int = 0> (float[2] A_in, float[2, 3] B_in, float[3] C_in) => (float[3] O_func) {
            sa = Constant<value: tensor = int64[2] {1, 2}>()
            A = Reshape(A_in, sa)
            M = Gemm<alpha=1.0, beta=1.0, transB = @transB>(A, B_in, C_in)
            sc = Constant<value: tensor = int64[1] {3}>()
            O_func = Reshape(M, sc)
        }
        """
        model = ir.from_onnx_text(model_text)
        func = next(iter(model.functions.values()))
        func.graph.inputs[0].shape = ir.Shape([2])
        func.graph.inputs[1].shape = ir.Shape([2, 3])
        func.graph.inputs[2].shape = ir.Shape([3])
        count = _gemm_to_matmul_add.gemm_to_matmul_add_rule.apply_to_model(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
