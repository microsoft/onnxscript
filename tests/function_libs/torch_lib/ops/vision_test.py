# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import torch
import torchvision
from torch.onnx import export
import os

class VisionOperatorTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "roi_align_test.onnx"

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_roi_align_export_with_seven_arguments(self):
        """
        Tests that torchvision::roi_align exports correctly with 7 positional arguments.
        This covers the signature change where output_size is decomposed into 
        pooled_height and pooled_width.
        """
        class RoiAlignModel(torch.nn.Module):
            def forward(self, x, boxes):
                return torchvision.ops.roi_align(
                    x, 
                    boxes, 
                    output_size=(7, 7), 
                    spatial_scale=0.5, 
                    sampling_ratio=2, 
                    aligned=True
                )

        # Create dummy inputs: (N, C, H, W) and (K, 5)
        x = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        boxes = torch.tensor([[0, 0, 0, 10, 10]], dtype=torch.float32)
        model = RoiAlignModel().eval()

        try:
            export(model, (x, boxes), self.model_path)
            export_success = True
        except Exception as e:
            export_success = False
            self.fail(f"torch.onnx.export failed for roi_align: {e}")

        self.assertTrue(export_success)

if __name__ == "__main__":
    unittest.main()
