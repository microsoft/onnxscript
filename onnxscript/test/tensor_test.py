"""Unit tests for the tensor module."""

import unittest

import numpy as np

from onnxscript import tensor


class TestTensor(unittest.TestCase):
    def test_scalar_tensor_supports_python_range(self):
        x = tensor.Tensor(np.array(1))
        self.assertEqual(list(range(x)), [0])

    def test_scalar_tensor_supports_int_conversion(self):
        x = tensor.Tensor(np.array(1))
        self.assertEqual(int(x), 1)

    def test_multi_dimensional_tensor_errors_when_converted_to_int(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        with self.assertRaises(TypeError):
            int(x)

    def test_scalar_tensor_supports_bool_conversion(self):
        x = tensor.Tensor(np.array(1))
        self.assertEqual(bool(x), True)

    def test_multi_dimensional_tensor_errors_when_converted_to_float(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        with self.assertRaises(TypeError):
            float(x)

    def test_scalar_tensor_supports_float_conversion(self):
        x = tensor.Tensor(np.array(1))
        self.assertEqual(float(x), 1.0)

    def test_multi_dimensional_tensor_errors_when_converted_to_bool(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            bool(x)


if __name__ == "__main__":
    unittest.main()
