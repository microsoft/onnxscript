# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the _convenience module."""

import unittest

import numpy as np

from onnxscript.ir import _convenience


class ConvenienceTest(unittest.TestCase):
    def test_tensor_accepts_torch_tensor(self):
        import torch as some_random_name

        torch_tensor = some_random_name.tensor([1, 2, 3])
        tensor = _convenience.tensor(torch_tensor)
        np.testing.assert_array_equal(tensor, torch_tensor.numpy())


if __name__ == "__main__":
    unittest.main()
