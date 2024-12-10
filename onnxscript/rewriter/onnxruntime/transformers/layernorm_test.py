# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from tests.common import testutils


class LNParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_ln_llama2(self):
        testutils.test_onnxruntime_rewrite(
            "ln_llama2", 4, {("", "SimplifiedLayerNormalization", "")}
        )


    def test_onnxruntime_rewrite_missing_optype(self):
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "ln_llama2", 1, {("fake_domain", "FakeOpType", "")}
            )


    @testutils.skip_if_no_cuda("CUDA is required for this test.")
    def test_skip_if_no_cuda(self):
        # This test will be skipped if CUDA is not available
        self.assertTrue(torch.cuda.is_available() and onnxruntime.get_device() == "GPU")


if __name__ == "__main__":
    unittest.main()
