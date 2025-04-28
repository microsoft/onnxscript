# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._core import fuse_xformers
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.ort_fusions.models._smollm_1 import smollm_test_1


class TestFuseXformers(unittest.TestCase):
    def test_fuse_xformers(self):
        test = smollm_test_1()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        model, fusion_count = fuse_xformers(model)

        # Check if the number of fusions applied for each fusion is correct
        self.assertEqual(fusion_count["rms_normalization"], 3)
        self.assertEqual(fusion_count["skip_layer_normalization"], 0)
        self.assertEqual(fusion_count["skip_rms_normalization"], 2)
        self.assertEqual(fusion_count["rotary_embedding"], 2)
        self.assertEqual(fusion_count["partial_rotary_embedding"], 0)
        self.assertEqual(fusion_count["cos_sin_cache"], 2)
        self.assertEqual(fusion_count["sdpa"], 1)
        self.assertEqual(fusion_count["mha"], 0)
        self.assertEqual(fusion_count["attention"], 0)
        self.assertEqual(fusion_count["gqa"], 0)
        self.assertEqual(fusion_count["gelu"], 0)

        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
