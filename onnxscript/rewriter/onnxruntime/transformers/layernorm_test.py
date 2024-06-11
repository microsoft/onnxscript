# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript.testing.common import testutils


class LNParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_ln_llama2(self):
        testutils.test_onnxruntime_rewrite(
            "ln_llama2", 4, {("", "SimplifiedLayerNormalization", "")}
        )


if __name__ == "__main__":
    unittest.main()
