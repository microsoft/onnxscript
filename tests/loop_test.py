import unittest

import numpy as np

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tests.common import testutils


class LoopOpTest(testutils.TestBase):
    def test_loop(self):
        """Basic loop test."""

        @script()
        def sumprod(x: FLOAT["N"], N: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod

        self.validate(sumprod)
        x = np.array([2])
        M = 3
        sum, prod = sumprod(x, M)
        self.assertEqual(sum, np.array([8]))
        self.assertEqual(prod, np.array([16]))

    def test_loop_bound(self):
        """Test with an expression for loop bound."""

        @script()
        def sumprod(x: FLOAT["N"], N: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(2 * N + 1):
                sum = sum + x
                prod = prod * x
            return sum, prod

        self.validate(sumprod)


if __name__ == "__main__":
    unittest.main()
