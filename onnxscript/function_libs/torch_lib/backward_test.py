# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=not-callable, unbalanced-tuple-unpacking

import copy
import sys
import unittest

import torch

import onnxscript.tools.training_helper
import onnxscript.tools.transformers_models
import onnxscript.tools.transformers_models.llama
from onnxscript._internal.version_utils import has_transformers, torch_older_than


class TestBackward(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.4"), reason="fails to export")
    def test_backward_working(self):
        class SimpleCNNN(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.fc1 = torch.nn.Linear(14, 10)

            def forward(self, x):
                return torch.nn.functional.relu(self.fc1(x))

        input_tensors = (torch.randn(1, 1, 14, 14),)
        model = SimpleCNNN()
        local_aot_ort = onnxscript.tools.training_helper.make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        expected_results, expected_gradients = onnxscript.tools.training_helper.train_loop(
            model, *input_tensors
        )
        results, gradients, onnx_models = onnxscript.tools.training_helper.train_loop(
            compiled_model,
            *input_tensors,
            dump_onnx_models=True,
            dump_prefix="_dump_testbw_working",
            dump_clean_first=True,
        )
        torch.testing.assert_allclose(expected_results[0], results[0], atol=1e-5, rtol=1e-5)

        # Checking there is only two generated graphs otherwise, it means there are graph breaks.
        self.assertEqual(len(onnx_models), 2)
        torch.testing.assert_allclose(
            expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5
        )

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.4"), reason="fails to export")
    @unittest.skipIf(True, reason="aten.conv_backward not implemented yet.")
    def test_backward_conv(self):
        class SimpleCNNN(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(
                    in_channels=1, out_channels=2, kernel_size=3, padding=1
                )
                self.fc1 = torch.nn.Linear(14, 10)

            def forward(self, x):
                y = torch.nn.functional.relu(self.conv1(x))
                z = self.fc1(y)
                return z

        input_tensors = (torch.randn(1, 1, 14, 14),)
        model = SimpleCNNN()
        local_aot_ort = onnxscript.tools.training_helper.make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        expected_results, expected_gradients = onnxscript.tools.training_helper.train_loop(
            model, *input_tensors
        )
        results, gradients, onnx_models = onnxscript.tools.training_helper.train_loop(
            compiled_model,
            *input_tensors,
            dump_onnx_models=True,
            dump_prefix="_dump_testbw_conv",
            dump_clean_first=True,
        )
        torch.testing.assert_allclose(expected_results[0], results[0], atol=1e-5, rtol=1e-5)

        # Checking there is only two generated graphs otherwise, it means there are graph breaks.
        self.assertEqual(len(onnx_models), 2)
        torch.testing.assert_allclose(
            expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
