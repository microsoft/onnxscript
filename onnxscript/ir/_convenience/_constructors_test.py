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

    def test_initializer(self):
        tensor = ir.Tensor(np.array([1, 2, 3]), dtype=ir.DataType.INT64, name="test_tensor")
        value = _constructors.initializer(tensor)
        self.assertEqual(value.const_value, tensor)
        self.assertEqual(value.name, tensor.name)
        self.assertEqual(value.shape, tensor.shape)
        self.assertEqual(value.dtype, tensor.dtype)


if __name__ == "__main__":
    unittest.main()
