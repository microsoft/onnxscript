# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import parameterized

from onnxscript.ir import _type_casting
import ml_dtypes


class TypeCastingTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_int4_even_sized_array(self, _: str, dtype):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x65, 0x87], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_int4_odd_sized_array(self, _: str, dtype):
        array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_int4_returns_flatten_array(self, _: str, dtype):
        array = np.array([[[1, 2, 3, 4, 5]]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)

    def test_unpack_uint4_with_padding(self):
        packed_data = np.array([0x21, 0x43, 0x65], dtype=np.uint8)
        expected = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        actual = _type_casting._unpack_uint4_as_uint8(packed_data, (5,))
        np.testing.assert_array_equal(actual, expected)


    def test_unpack_float4e2m1(self):
        packed_data = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)
        expected_shape = (2, 4)
        actual = _type_casting.unpack_float4e2m1(packed_data, expected_shape)
        self.assertEqual(actual.shape, expected_shape)


    def test_unpack_uint4(self):
        packed_data = np.array([0x21, 0x43, 0x65, 0x87], dtype=np.uint8)
        expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=ml_dtypes.uint4)
        actual = _type_casting.unpack_uint4(packed_data, (2, 4))
        np.testing.assert_array_equal(actual, expected)



if __name__ == "__main__":
    unittest.main(verbosity=2)
