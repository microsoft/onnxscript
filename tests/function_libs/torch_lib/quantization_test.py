# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test quantized model export."""

from __future__ import annotations

import unittest

import onnx
import torch
import torch._export as torch_export
from torch.ao.quantization import quantize_pt2e
from torch.ao.quantization.quantizer import xnnpack_quantizer

from onnxscript._internal import version_utils

import warnings
import unittest.mock

class QuantizedModelExportTest(unittest.TestCase):
    @unittest.skipIf(
        version_utils.torch_older_than("2.4"),
        "Dynamo exporter fails at the modularization step.",
    )
    def test_simple_quantized_model(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.randn(1, 5),)
        model = TestModel().eval()

        # Step 1. program capture
        pt2e_torch_model = torch_export.capture_pre_autograd_graph(model, example_inputs)

        # Step 2. quantization
        quantizer = xnnpack_quantizer.XNNPACKQuantizer().set_global(
            xnnpack_quantizer.get_symmetric_quantization_config()
        )
        pt2e_torch_model = quantize_pt2e.prepare_pt2e(pt2e_torch_model, quantizer)

        # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
        pt2e_torch_model(*example_inputs)

        # Convert the prepared model to a quantized model
        pt2e_torch_model = quantize_pt2e.convert_pt2e(pt2e_torch_model, fold_quantize=False)
        program = torch.onnx.dynamo_export(pt2e_torch_model, *example_inputs)
        onnx.checker.check_model(program.model_proto, full_check=True)

    def test_is_onnxruntime_training_with_push_back_batch(self):
        with unittest.mock.patch('onnxruntime.training', create=True):
            mock_ortvaluevector = type('OrtValueVector', (object,), {'push_back_batch': True})()
            with unittest.mock.patch('onnxruntime.capi.onnxruntime_pybind11_state.OrtValueVector', mock_ortvaluevector):
                result = version_utils.is_onnxruntime_training()
                self.assertTrue(result)


    def test_ignore_warnings_suppresses_warning(self):
        @version_utils.ignore_warnings(UserWarning)
        def dummy_function(self):
            warnings.warn("This is a user warning", UserWarning)
            return True
    
        result = dummy_function(self)
        self.assertTrue(result)


    def test_has_transformers_not_installed(self):
        with unittest.mock.patch.dict('sys.modules', {'transformers': None}):
            result = version_utils.has_transformers()
            self.assertFalse(result)


    def test_has_transformers_installed(self):
        with unittest.mock.patch('sys.modules', {'transformers': unittest.mock.Mock()}):
            result = version_utils.has_transformers()
            self.assertTrue(result)


    def test_numpy_older_than_true(self):
        with unittest.mock.patch('numpy.__version__', '1.18.0'):
            result = version_utils.numpy_older_than("1.19.0")
            self.assertTrue(result)


    def test_onnxruntime_older_than_true(self):
        with unittest.mock.patch('onnxruntime.__version__', '1.8.0'):
            result = version_utils.onnxruntime_older_than("1.9.0")
            self.assertTrue(result)


    def test_ignore_warnings_raises_assertion(self):
        with self.assertRaises(AssertionError):
            @version_utils.ignore_warnings(None)
            def dummy_function():
                pass


    def test_is_onnxruntime_training_no_training(self):
        with unittest.mock.patch.dict('sys.modules', {'onnxruntime.training': None}):
            result = version_utils.is_onnxruntime_training()
            self.assertFalse(result)


    def test_transformers_older_than_no_transformers(self):
        with unittest.mock.patch.dict('sys.modules', {'transformers': None}):
            result = version_utils.transformers_older_than("4.0")
            self.assertIsNone(result)



if __name__ == "__main__":
    unittest.main()
