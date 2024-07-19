"""Test quantized model export."""

from __future__ import annotations

import onnx
import torch
import unittest
from torch.ao.quantization import quantize_pt2e
import torch._export
from torch.ao.quantization.quantizer import xnnpack_quantizer

class QuantizedModelExportTest(unittest.TestCase):
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
        pt2e_torch_model = torch._export.capture_pre_autograd_graph(model, example_inputs)

        # Step 2. quantization
        quantizer = xnnpack_quantizer.XNNPACKQuantizer().set_global(xnnpack_quantizer.get_symmetric_quantization_config())
        pt2e_torch_model = quantize_pt2e.prepare_pt2e(pt2e_torch_model, quantizer)

        # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
        pt2e_torch_model(*example_inputs)

        # Convert the prepared model to a quantized model
        pt2e_torch_model = quantize_pt2e.convert_pt2e(pt2e_torch_model, fold_quantize=False)
        program = torch.onnx.dynamo_export(pt2e_torch_model, *example_inputs)
        onnx.checker.check_model(program.model_proto, full_check=True)


if __name__ == "__main__":
    unittest.main()
