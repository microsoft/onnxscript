from __future__ import annotations

import unittest

import numpy as np

from tests.common import testutils


class BiasSplitGeluParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_gelu_phi_1_5(self):
        testutils.test_onnxruntime_rewrite(
            "geglu_stable_diffusion_unet", 4, {("com.microsoft", "BiasSplitGelu", "")}
        )


if __name__ == "__main__":
    unittest.main()
