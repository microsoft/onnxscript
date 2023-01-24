"""Unit tests for the tensor module."""

import unittest

import numpy as np

from onnxscript import tensor


class TestTensor(unittest.TestCase):
    def test_scalar_tensor_supports_python_range(self):
        x = tensor.Tensor(np.array(3))
        self.assertEqual(range(x), range(3))

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

    def test_it_can_be_initialized_with_none(self):
        x = tensor.Tensor(None)
        with self.assertRaises(ValueError):
            _ = x.value

    def test_rank_is_zero_for_scalar(self):
        x = tensor.Tensor(np.array(1))
        self.assertEqual(x.rank, 0)

    def test_rank_is_one_for_vector(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        self.assertEqual(x.rank, 1)

    def test_rank_is_two_for_matrix(self):
        x = tensor.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(x.rank, 2)


if __name__ == "__main__":
    unittest.main()
