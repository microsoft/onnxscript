# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript.testing.common import testutils


class MHAParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_llama2_4_34(self):
        testutils.test_onnxruntime_rewrite(
            "attn_llama2_4_34", 2, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_llama2_4_36(self):
        testutils.test_onnxruntime_rewrite(
            "attn_llama2_4_36", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_attn_yi_4_37(self):
        testutils.test_onnxruntime_rewrite(
            "attn_yi_4_37", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_sdpa_llama2_4_36(self):
        # TODO: Clean-up naming logic of test models.
        # Package version was not considered.
        testutils.test_onnxruntime_rewrite(
            "sdpa_llama2", 4, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @unittest.skip("TODO: Fails parity check")
    def test_sdpa_llama2_4_38(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_llama2_4_38", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("GQA Kernel unsupported on CPU.")
    def test_sdpa_yi_4_36(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_yi", 2, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @unittest.skip("TODO: Fails parity check")
    def test_sdpa_yi_4_38(self):
        testutils.test_onnxruntime_rewrite(
            "sdpa_yi_4_38", 1, {("com.microsoft", "GroupQueryAttention", "")}
        )

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_stable_diffusion_unet(self):
        testutils.test_onnxruntime_rewrite(
            "attn_stable_diffusion_unet", 2, {("com.microsoft", "MultiHeadAttention", "")}
        )


class AttnParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_phi_1_5(self):
        testutils.test_onnxruntime_rewrite(
            "attn_phi_1_5", 4, {("com.microsoft", "Attention", "")}
        )

    @testutils.skip_if_no_cuda("CPU has parity issue.")
    def test_attn_stable_diffusion_unet_without_encoder_hidden_states(self):
        testutils.test_onnxruntime_rewrite(
            "attn_stable_diffusion_unet_without_encoder_hidden_states",
            2,
            {("com.microsoft", "Attention", "")},
        )


if __name__ == "__main__":
    unittest.main()
