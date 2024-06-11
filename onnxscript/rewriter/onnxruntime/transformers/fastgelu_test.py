# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript.testing.common import testutils


class FastGeluParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_gelu_phi_1_5(self):
        testutils.test_onnxruntime_rewrite(
            "gelu_phi_1_5", 4, {("com.microsoft", "FastGelu", "")}
        )


if __name__ == "__main__":
    unittest.main()
