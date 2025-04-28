# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.ort_fusions.models._smollm_1 import smollm_test_1
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization


class TestRmsNormalization(unittest.TestCase):
    def test_smollm(self):
        smollm_test = smollm_test_1()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = smollm_test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        fuse_rms_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SimplifiedLayerNormalization", op_types)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
