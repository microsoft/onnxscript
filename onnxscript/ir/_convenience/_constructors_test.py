# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the _constructors module."""

import unittest

import numpy as np

from onnxscript import ir
from onnxscript.ir._convenience import _constructors


class ConstructorsTest(unittest.TestCase):
    def test_tensor_accepts_torch_tensor(self):
        import torch as some_random_name  # pylint: disable=import-outside-toplevel

        torch_tensor = some_random_name.tensor([1, 2, 3])
        tensor = _constructors.tensor(torch_tensor)
        np.testing.assert_array_equal(tensor, torch_tensor.numpy())

    def test_tensor_raises_value_error_for_empty_sequence_without_dtype(self):
        with self.assertRaises(ValueError):
            _constructors.tensor([])

    def test_tensor_handles_empty_sequence_with_dtype(self):
        tensor = _constructors.tensor([], dtype=ir.DataType.FLOAT)
        np.testing.assert_array_equal(tensor.numpy(), np.array([], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
