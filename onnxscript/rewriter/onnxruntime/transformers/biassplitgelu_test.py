# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript.testing import testutils


class BiasSplitGeluParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @testutils.skip_if_no_cuda("BiasSplitGelu Kernel unsupported on CPU.")
    def test_geglu_stable_diffusion_unet(self):
        testutils.test_onnxruntime_rewrite(
            "geglu_stable_diffusion_unet", 4, {("com.microsoft", "BiasSplitGelu", "")}
        )


if __name__ == "__main__":
    unittest.main()
