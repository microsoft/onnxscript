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

    @parameterized.parameterized.expand(
        [
            ("negative_infinity", np.uint16(0b1_11111111_0000000)),
            ("negative_min_normal", np.uint16(0b1_11111110_1111111)),
            ("negative_max_normal", np.uint16(0b1_00000001_0000000)),
            ("negative_min_subnormal", np.uint16(0b1_00000000_1111111)),
            ("negative_max_subnormal", np.uint16(0b1_00000000_0000001)),
            ("negative_zero", np.uint16(0b1_00000000_0000000)),
            ("positive_zero", np.uint16(0b0_00000000_0000000)),
            ("positive_min_subnormal", np.uint16(0b0_00000000_0000001)),
            ("positive_max_subnormal", np.uint16(0b0_00000000_1111111)),
            ("positive_min_normal", np.uint16(0b0_00000001_0000000)),
            ("positive_max_normal", np.uint16(0b0_11111110_1111111)),
            ("positive_infinity", np.uint16(0b0_11111111_0000000)),
            ("positive_nan", np.uint16(0b0_11111111_1000000)),
            ("positive_one", np.uint16(0b0_00111111_0000000)),
            ("negative_one", np.uint16(0b1_00111111_0000000)),
        ]
    )
    def test_float32_to_bfloat16(self, _: str, binary: np.uint16):
        value = np.array([binary << 16]).astype(np.uint32).view(np.float32)
        expected = np.array([binary])
        actual = _type_casting.float32_to_bfloat16(value)
        np.testing.assert_array_equal(actual, expected)

    def test_float32_to_float8e5m2(self):
        array = np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        _type_casting.float32_to_float8e5m2(array)

    def test_float32_to_float8e5m2fnuz(self):
        array = np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        _type_casting.float32_to_float8e5m2fnuz(array)

    def test_float32_to_float8e4m3fn(self):
        array = np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        _type_casting.float32_to_float8e4m3fn(array)

    def test_float32_to_float8e4m3fnuz(self):
        array = np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        _type_casting.float32_to_float8e4m3fnuz(array)


if __name__ == "__main__":
    unittest.main(verbosity=2)
