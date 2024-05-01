import unittest

import numpy as np
import parameterized

from onnxscript.ir import _type_casting


class TypeCastingTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("signed", np.float32),
            ("unsigned", np.uint32),
        ]
    )
    def test_pack_int4_even_sized_array(self, _: str, dtype):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x65, 0x87], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.float32),
            ("unsigned", np.uint32),
        ]
    )
    def test_pack_int4_odd_sized_array(self, _: str, dtype):
        array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.float32),
            ("unsigned", np.uint32),
        ]
    )
    def test_pack_int4_returns_flatten_array(self, _: str, dtype):
        array = np.array([[[1, 2, 3, 4, 5]]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_int4(array)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
