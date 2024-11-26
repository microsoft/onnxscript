# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from tests.common import testutils
import unittest.mock
import onnx
import onnx


class MHAParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_llama2_4_34(self):
        testutils.test_onnxruntime_rewrite(
            "attn_llama2_4_34", 2, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_llama2_4_36(self):
        testutils.test_onnxruntime_rewrite(
            "attn_llama2_4_36", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_yi_4_37(self):
        testutils.test_onnxruntime_rewrite(
            "attn_yi_4_37", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_sdpa_llama2_4_36(self):
        # TODO: Clean-up naming logic of test models.
        # Package version was not considered.
        testutils.test_onnxruntime_rewrite(
            "sdpa_llama2", 4, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @unittest.skip("TODO: Fails parity check")
    def test_sdpa_llama2_4_38(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_llama2_4_38", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_sdpa_yi_4_36(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_yi", 2, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @unittest.skip("TODO: Fails parity check")
    def test_sdpa_yi_4_38(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_yi_4_38", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_stable_diffusion_unet(self):
        testutils.test_onnxruntime_rewrite(
            "attn_stable_diffusion_unet", 2, {("com.microsoft", "MultiHeadAttention", "")}
        )


class AttnParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_phi_1_5(self):
        testutils.test_onnxruntime_rewrite(
            "attn_phi_1_5", 4, {("com.microsoft", "Attention", "")}
        )

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_stable_diffusion_unet_without_encoder_hidden_states(self):
        testutils.test_onnxruntime_rewrite(
            "attn_stable_diffusion_unet_without_encoder_hidden_states",
            2,
            {("com.microsoft", "Attention", "")},
        )


    @unittest.mock.patch('onnxruntime.InferenceSession')
    @unittest.mock.patch('onnx.load')
    @unittest.mock.patch('tests.common.testutils.evaluation_utils.load_test_data', return_value=({"input": np.array([1.0])}, [np.array([1.0, 2.0])]))
    def test_onnxruntime_rewrite_output_shape_mismatch(self, mock_load_test_data, mock_load, mock_inference_session):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=[
                    onnx.helper.make_node("Add", ["X", "Y"], ["Z"], domain=""),
                ],
                name="test_graph",
                inputs=[
                    onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1]),
                ],
            )
        )
        mock_load.return_value = model
        mock_session = mock_inference_session.return_value
        mock_session.run.return_value = [np.array([1.0])]
        
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "dummy_model", 1, {("", "Add", "")}
            )


    def test_validate_method(self):
        class MockFunction:
            def to_function_proto(self):
                return "function_proto"
        
        test_base = testutils.TestBase()
        result = test_base.validate(MockFunction())
        self.assertEqual(result, "function_proto")


    @unittest.mock.patch('onnxruntime.InferenceSession')
    @unittest.mock.patch('onnx.load')
    @unittest.mock.patch('tests.common.testutils.evaluation_utils.load_test_data', return_value=({}, []))
    def test_onnxruntime_rewrite_success(self, mock_load_test_data, mock_load, mock_inference_session):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=[
                    onnx.helper.make_node("Add", ["X", "Y"], ["Z"], domain=""),
                    onnx.helper.make_node("Relu", ["Z"], ["W"], domain=""),
                ],
                name="test_graph",
                inputs=[
                    onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [1]),
                ],
            )
        )
        mock_load.return_value = model
        testutils.test_onnxruntime_rewrite(
            "dummy_model", 1, {("", "Add", ""), ("", "Relu", "")}
        )


    @unittest.mock.patch('onnxruntime.InferenceSession')
    @unittest.mock.patch('onnx.load')
    @unittest.mock.patch('tests.common.testutils.evaluation_utils.load_test_data', return_value=({}, []))
    def test_onnxruntime_rewrite_missing_optypes(self, mock_load_test_data, mock_load, mock_inference_session):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=[
                    onnx.helper.make_node("Add", ["X", "Y"], ["Z"], domain=""),
                ],
                name="test_graph",
                inputs=[
                    onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1]),
                ],
            )
        )
        mock_load.return_value = model
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "dummy_model", 1, {("", "Relu", "")}
            )


    def test_op_type_analysis_visitor(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes=[
                    onnx.helper.make_node("Add", ["X", "Y"], ["Z"], domain=""),
                    onnx.helper.make_node("Relu", ["Z"], ["W"], domain=""),
                ],
                name="test_graph",
                inputs=[
                    onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1]),
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [1]),
                ],
            )
        )
        visitor = testutils.OpTypeAnalysisVisitor()
        visitor.visit_model(model)
        expected_op_types = {("", "Add", ""), ("", "Relu", "")}
        self.assertEqual(visitor.op_types, expected_op_types)


if __name__ == "__main__":
    unittest.main()
