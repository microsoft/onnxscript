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


if __name__ == "__main__":
    unittest.main()
