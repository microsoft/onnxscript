# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._core import fuse_xformers
from onnxscript.rewriter.ort_fusions._smollm_1 import smollm_test_1
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run


class TestFuseXformers(unittest.TestCase):
    def test_fuse_xformers(self):
        test = smollm_test_1()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        model = fuse_xformers(model)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
