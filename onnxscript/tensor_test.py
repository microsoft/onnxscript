# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for the tensor module."""

import unittest

import numpy as np

from onnxscript import tensor


class TestTensor(unittest.TestCase):
    def _check_values_and_shape(self, result, elements, shape):
        values = list(result.value.flatten())
        self.assertEqual(values, elements)
        self.assertEqual(result.shape, shape)

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

    def test_getitem_index(self):
        # Create tensor:
        # [ [0, 1, 2],
        #   [3, 4, 5] ]
        data = np.array(range(6), dtype=np.int32).reshape(2, 3)
        x = tensor.Tensor(data)

        y = x[0]
        # Should return first row: [0, 1, 2]
        self._check_values_and_shape(y, [0, 1, 2], (3,))

        y = x[:, 0]
        # Should return first column: [0, 3]
        self._check_values_and_shape(y, [0, 3], (2,))

        y = x[:, -1]
        # Should return last column: [2, 5]
        self._check_values_and_shape(y, [2, 5], (2,))

        y = x[0, 0]
        # Should return first element (as a scalar)
        self._check_values_and_shape(y, [0], ())

    def test_getitem_slice(self):
        # Create tensor:
        # [ [0, 1, 2],
        #   [3, 4, 5],
        #   [6, 7, 8],
        #   [9, 10, 11] ]
        data = np.array(range(12), dtype=np.int32).reshape(4, 3)
        x = tensor.Tensor(data)

        y = x[0:2]
        # Should return first two rows
        self._check_values_and_shape(y, [0, 1, 2, 3, 4, 5], (2, 3))

        y = x[:, 0:2]
        # Should return first two columns
        self._check_values_and_shape(y, [0, 1, 3, 4, 6, 7, 9, 10], (4, 2))

        y = x[0:2, 0:2]
        # Should return first 2x2 sub-matrix
        self._check_values_and_shape(y, [0, 1, 3, 4], (2, 2))

    def test_getitem_index_and_slice(self):
        # Create tensor:
        # [ [0, 1, 2],
        #   [3, 4, 5],
        #   [6, 7, 8],
        #   [9, 10, 11] ]
        data = np.array(range(12), dtype=np.int32).reshape(4, 3)
        x = tensor.Tensor(data)

        y = x[0:2, 0]
        # Should return first two rows, first column
        self._check_values_and_shape(y, [0, 3], (2,))

        y = x[0, 0:2]
        # Should return first row, first 2 columns:
        self._check_values_and_shape(y, [0, 1], (2,))

    def test_getitem_gather(self):
        # Create tensor:
        # [ [0, 1, 2],
        #   [3, 4, 5],
        #   [6, 7, 8],
        #   [9, 10, 11] ]
        data = np.array(range(12), dtype=np.int32).reshape(4, 3)
        x = tensor.Tensor(data)

        indices_0_and_3 = tensor.Tensor(np.array([0, 3], dtype=np.int32))
        y = x[indices_0_and_3]
        # Should return row 0 and row 3
        self._check_values_and_shape(y, [0, 1, 2, 9, 10, 11], (2, 3))

        index_0 = tensor.Tensor(np.array([0], dtype=np.int32))
        y = x[index_0]
        # Should return row 0, but of shape 1x3
        self._check_values_and_shape(y, [0, 1, 2], (1, 3))

        y = x[indices_0_and_3, index_0]
        # Should return submatrix consisting of rows 0 and 3, column 0
        self._check_values_and_shape(y, [0, 9], (2, 1))


    def test_tensor_reverse_matmul(self):
        x = tensor.Tensor(np.array([[1, 2], [3, 4]]))
        y = tensor.Tensor(np.array([[5, 6], [7, 8]]))
        result = y @ x
        expected = np.array([[23, 34], [31, 46]])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_equality(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([1, 2, 3]))
        result = x == y
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_reverse_sub_with_tensor(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([10, 20, 30]))
        result = y - x
        expected = np.array([9, 18, 27])
        np.testing.assert_array_equal(result.value, expected)


    def test_getitem_negative_step_slice(self):
        data = np.array(range(12), dtype=np.int32).reshape(4, 3)
        x = tensor.Tensor(data)
        y = x[::-1]
        expected = np.array([[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
        np.testing.assert_array_equal(y.value, expected)


    def test_tensor_greater_than_or_equal(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([2, 2, 2]))
        result = x >= y
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_less_than_or_equal(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([2, 2, 4]))
        result = x <= y
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_reverse_sub(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = 10
        result = y - x
        expected = np.array([9, 8, 7])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_greater_than(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([0, 2, 2]))
        result = x > y
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_less_than(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([2, 2, 4]))
        result = x < y
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_reverse_mul_with_tensor(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([2, 3, 4]))
        result = y * x
        expected = np.array([2, 6, 12])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_truediv(self):
        x = tensor.Tensor(np.array([4.0, 9.0, 16.0]))
        y = tensor.Tensor(np.array([2.0, 3.0, 4.0]))
        result = x / y
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result.value, expected)


    def test_tensor_pow(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = tensor.Tensor(np.array([2, 3, 4]))
        result = x ** y
        expected = np.array([1, 8, 81])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_or(self):
        x = tensor.Tensor(np.array([True, False, True]))
        y = tensor.Tensor(np.array([True, True, False]))
        result = x | y
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_and(self):
        x = tensor.Tensor(np.array([True, False, True]))
        y = tensor.Tensor(np.array([True, True, False]))
        result = x & y
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_reverse_mul(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = 2
        result = y * x
        expected = np.array([2, 4, 6])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_reverse_add(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        y = 10
        result = y + x
        expected = np.array([11, 12, 13])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_mod_integer(self):
        x = tensor.Tensor(np.array([5, 6], dtype=np.int32))
        y = tensor.Tensor(np.array([2, 2], dtype=np.int32))
        result = x % y
        expected = np.array([1, 0], dtype=np.int32)
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_negation(self):
        x = tensor.Tensor(np.array([1, -2, 3]))
        result = -x
        expected = np.array([-1, 2, -3])
        np.testing.assert_array_equal(result.value, expected)


    def test_tensor_not_equal_scalar(self):
        x = tensor.Tensor(np.array(1))
        y = tensor.Tensor(np.array(2))
        result = x != y
        self.assertTrue(result.value)


    def test_getitem_unexpected_index_type(self):
        data = np.array([1, 2, 3])
        x = tensor.Tensor(data)
        with self.assertRaises(TypeError):
            _ = x["invalid"]


    def test_getitem_index_exceeds_rank(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        x = tensor.Tensor(data)
        with self.assertRaises(ValueError):
            _ = x[0, 0, 0]


    def test_tensor_mod_floating_point(self):
        x = tensor.Tensor(np.array([5.5, 6.5], dtype=np.float32))
        y = tensor.Tensor(np.array([2.0, 2.0], dtype=np.float32))
        result = x % y
        expected = np.array([1.5, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.value, expected)


    def test_getitem_opset_version_error(self):
        data = np.array([1, 2, 3])
        x = tensor.Tensor(data, opset=type('Opset', (object,), {'version': 12})())
        with self.assertRaises(RuntimeError):
            _ = x[0]


    def test_tensor_repr(self):
        x = tensor.Tensor(np.array([1, 2, 3]))
        self.assertEqual(repr(x), "Tensor(array([1, 2, 3]))")


    def test_tensor_initialization_with_invalid_type(self):
        with self.assertRaises(TypeError):
            tensor.Tensor([1, 2, 3])  # Not a numpy array


if __name__ == "__main__":
    unittest.main()
