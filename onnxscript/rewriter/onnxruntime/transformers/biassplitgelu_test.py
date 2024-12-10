# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from tests.common import testutils

import unittest.mock
import onnx
import onnx

class BiasSplitGeluParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("BiasSplitGelu Kernel unsupported on CPU.")
    def test_geglu_stable_diffusion_unet(self):
        testutils.test_onnxruntime_rewrite(
            "geglu_stable_diffusion_unet", 4, {("com.microsoft", "BiasSplitGelu", "")}
        )


    def test_validate_method_calls_to_function_proto(self):
        class MockFunction:
            def to_function_proto(self):
                return "FunctionProtoCalled"
    
        test_base = testutils.TestBase()
        result = test_base.validate(MockFunction())
        self.assertEqual(result, "FunctionProtoCalled")


    @unittest.mock.patch('onnx.load')
    @unittest.mock.patch('tests.common.testutils.evaluation_utils.load_test_data')
    @unittest.mock.patch('tests.common.testutils.optimizer.optimize')
    @unittest.mock.patch('tests.common.testutils.ort_rewriter.rewrite')
    @unittest.mock.patch('onnxruntime.InferenceSession')
    def test_onnxruntime_rewrite_output_mismatch(self, mock_inference_session, mock_rewrite, mock_optimize, mock_load_test_data, mock_onnx_load):
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
        mock_onnx_load.return_value = model
        mock_load_test_data.return_value = ({}, [np.array([1.0])])
        mock_optimize.return_value = model
        mock_rewrite.return_value = model
        mock_session_instance = mock_inference_session.return_value
        mock_session_instance.run.return_value = [np.array([2.0])]
    
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "mock_model", 1, {("", "Add", "")}
            )


    @unittest.mock.patch('torch.cuda.is_available', return_value=True)
    @unittest.mock.patch('onnxruntime.get_device', return_value='GPU')
    def test_skip_if_no_cuda_available(self, mock_cuda, mock_device):
        @testutils.skip_if_no_cuda("Test reason")
        def dummy_test(self):
            return "Test Passed"
    
        result = dummy_test(self)
        self.assertEqual(result, "Test Passed")


    @unittest.mock.patch('onnx.load')
    @unittest.mock.patch('tests.common.testutils.evaluation_utils.load_test_data')
    @unittest.mock.patch('tests.common.testutils.optimizer.optimize')
    @unittest.mock.patch('tests.common.testutils.ort_rewriter.rewrite')
    def test_onnxruntime_rewrite_missing_optype(self, mock_rewrite, mock_optimize, mock_load_test_data, mock_onnx_load):
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
        mock_onnx_load.return_value = model
        mock_load_test_data.return_value = ({}, [])
        mock_optimize.return_value = model
        mock_rewrite.return_value = model
    
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "mock_model", 1, {("com.microsoft", "NonExistentOp", "")}
            )


    def test_op_type_analysis_visitor(self):
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
        visitor = testutils.OpTypeAnalysisVisitor()
        visitor.visit_model(model)
        self.assertIn(("", "Add", ""), visitor.op_types)


if __name__ == "__main__":
    unittest.main()
