# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test display() methods in various classes."""

import contextlib
import unittest

import numpy as np

import onnxscript.ir as ir


class DisplayTest(unittest.TestCase):
    def test_tensor_display_does_not_raise_on_nan_values(self):
        array_with_nan = np.array([np.inf, -np.inf, np.nan, 5, -10], dtype=np.float32)
        tensor = ir.Tensor(array_with_nan, dtype=ir.DataType.FLOAT)
        with contextlib.redirect_stdout(None):
            tensor.display()


if __name__ == "__main__":
    unittest.main()
