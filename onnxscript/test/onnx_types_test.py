# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# mypy: disable-error-code=misc

"""Unit tests for the onnx_types module."""
from __future__ import annotations

import unittest

from parameterized import parameterized

from onnxscript.onnx_types import DOUBLE, FLOAT, DType, TensorType, tensor_type_registry


class TestOnnxTypes(unittest.TestCase):
    def test_instantiation(self):
        with self.assertRaises(NotImplementedError):
            TensorType()
        with self.assertRaises(NotImplementedError):
            FLOAT()
        with self.assertRaises(NotImplementedError):
            FLOAT[...]()

    @parameterized.expand(tensor_type_registry.items())
    def test_type_properties(self, dtype: DType, tensor_type: type[TensorType]):
        self.assertEqual(tensor_type.dtype, dtype)
        self.assertIsNone(tensor_type.shape)
        self.assertEqual(tensor_type[...].shape, ...)  # type: ignore[index]
        self.assertEqual(tensor_type[...].dtype, dtype)  # type: ignore[index]
        self.assertEqual(tensor_type[1, 2, 3].shape, (1, 2, 3))  # type: ignore[index]
        self.assertEqual(tensor_type[1, 2, 3].dtype, dtype)  # type: ignore[index]

    @parameterized.expand([(dtype,) for dtype in tensor_type_registry])
    def test_dtype_bound_to_subclass(self, dtype: DType):
        with self.assertRaises(ValueError):
            type(f"InvalidTensorTypeSubclass_{dtype}", (TensorType,), {}, dtype=dtype)

    def test_shaped_doesnt_reshape(self):
        with self.assertRaises(ValueError):
            FLOAT[1][...]  # pylint: disable=pointless-statement

    @parameterized.expand(
        [
            (FLOAT, FLOAT),
            (FLOAT[None], FLOAT[None]),
            (FLOAT[1, 2, 3], FLOAT[1, 2, 3]),
            (FLOAT[1], FLOAT[1]),
            (FLOAT[...], FLOAT[Ellipsis]),
            (FLOAT["M"], FLOAT["M"]),
            (FLOAT["M", "N"], FLOAT["M", "N"]),
            (FLOAT["M", 3, 4], FLOAT["M", 3, 4]),
        ]
    )
    def test_shapes_are_same_type(self, a: TensorType, b: TensorType):
        self.assertIs(a, b)

    @parameterized.expand(
        [
            (FLOAT[0], FLOAT[None]),
            (FLOAT[1, 2], FLOAT[3, 4]),
            (FLOAT[2, 1], FLOAT[1, 2]),
            (FLOAT["M", "N"], FLOAT["N", "M"]),
            (FLOAT, DOUBLE),
            (FLOAT[1], DOUBLE[1]),
            (FLOAT["X"], DOUBLE["X"]),
        ]
    )
    def test_shapes_are_not_same_type(self, a: TensorType, b: TensorType):
        self.assertIsNot(a, b)


if __name__ == "__main__":
    unittest.main()
