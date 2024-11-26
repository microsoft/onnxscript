# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from tests.common import testutils


class FastGeluParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_gelu_phi_1_5(self):
        testutils.test_onnxruntime_rewrite(
            "gelu_phi_1_5", 4, {("com.microsoft", "FastGelu", "")}
        )


    def test_validate_method_calls_to_function_proto(self):
        class MockFunction:
            def to_function_proto(self):
                return "FunctionProtoCalled"
    
        test_base = testutils.TestBase()
        result = test_base.validate(MockFunction())
        self.assertEqual(result, "FunctionProtoCalled")


    def test_output_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "gelu_phi_1_5", 1, {("com.microsoft", "FastGelu", "")}, atol=0
            )


    def test_assertion_error_for_missing_optypes(self):
        with self.assertRaises(AssertionError):
            testutils.test_onnxruntime_rewrite(
                "gelu_phi_1_5", 1, {("com.microsoft", "NonExistentOpType", "")}
            )


    @testutils.skip_if_no_cuda("Testing skip if no CUDA.")
    def test_skip_if_no_cuda(self):
        self.fail("This test should be skipped if CUDA is not available.")


if __name__ == "__main__":
    unittest.main()
