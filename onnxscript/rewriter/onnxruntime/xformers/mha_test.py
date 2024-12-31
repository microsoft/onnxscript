# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
import onnxscript.rewriter.onnxruntime.xformers as xformers
from onnxscript.rewriter.onnxruntime.xformers._smollm_1layer import _SmollmTestData
from onnxscript.rewriter.onnxruntime.xformers._test_utils import assert_allclose, ort_run


class TestMultiHeadAttention(unittest.TestCase):
    def test_smollm(self):
        # Generate model
        smollm_test = _SmollmTestData()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        xformers.fuse_rms_normalization(model)
        xformers.fuse_normalization(model)
        xformers.fuse_rotary_embedding(model)
        xformers.fuse_cos_sin_cache(model)

        # Run model
        inputs = smollm_test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)

        # Fuse SDPA and MHA
        sdpa_count = xformers.fuse_sdpa(model)
        self.assertGreater(sdpa_count, 0)
        mha_count = xformers.fuse_mha(model)
        self.assertGreater(mha_count, 0)

        # Run model again
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
