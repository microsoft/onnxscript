# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.onnxruntime.xformers import fuse_cos_sin_cache, fuse_rotary_embedding
<<<<<<< HEAD
from onnxscript.rewriter.onnxruntime.xformers._smollm_1 import TestData
=======
from onnxscript.rewriter.onnxruntime.xformers._smollm_1layer import _SmollmTestData
>>>>>>> main
from onnxscript.rewriter.onnxruntime.xformers._test_utils import assert_allclose, ort_run


class TestCosSinCacheTransform(unittest.TestCase):
    def test_smollm(self):
<<<<<<< HEAD
        smollm_test = TestData()
=======
        smollm_test = _SmollmTestData()
>>>>>>> main
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = smollm_test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        count = fuse_rotary_embedding(model)
        self.assertGreater(count, 0)
        count = fuse_cos_sin_cache(model)
        self.assertGreater(count, 0)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
