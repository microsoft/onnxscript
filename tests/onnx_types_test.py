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

import onnx
from onnxscript.onnx_types import onnx_type_to_onnxscript_repr
from onnxscript.onnx_types import _check_dim
from onnxscript.onnx_types import DOUBLE, FLOAT, TensorType, tensor_type_registry


class TestOnnxTypes(unittest.TestCase):
    def test_instantiation(self):
        with self.assertRaises(NotImplementedError):
            TensorType()
        with self.assertRaises(NotImplementedError):
            FLOAT()
        with self.assertRaises(NotImplementedError):
            FLOAT[...]()

    @parameterized.expand(tensor_type_registry.items())
    def test_type_properties(self, dtype: int, tensor_type: type[TensorType]):
        self.assertEqual(tensor_type.dtype, dtype)
        self.assertIsNone(tensor_type.shape)
        self.assertEqual(tensor_type[...].shape, ...)  # type: ignore[index]
        self.assertEqual(tensor_type[...].dtype, dtype)  # type: ignore[index]
        self.assertEqual(tensor_type[1, 2, 3].shape, (1, 2, 3))  # type: ignore[index]
        self.assertEqual(tensor_type[1, 2, 3].dtype, dtype)  # type: ignore[index]

    @parameterized.expand([(dtype,) for dtype in tensor_type_registry])
    def test_dtype_bound_to_subclass(self, dtype: int):
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


    def test_to_type_proto_unsupported_onnx_type(self):
        class MockTypeProto:
            def HasField(self, field):
                return False
    
        type_proto = MockTypeProto()
        with self.assertRaises(NotImplementedError):
            onnx_type_to_onnxscript_repr(type_proto)


    def test_onnx_type_to_onnxscript_repr(self):
        # Mocking an ONNX TypeProto
        class MockDim:
            def __init__(self, dim_value=None, dim_param=None):
                self.dim_value = dim_value
                self.dim_param = dim_param
    
            def HasField(self, field):
                if field == "dim_value":
                    return self.dim_value is not None
                if field == "dim_param":
                    return self.dim_param is not None
                return False
    
        class MockShape:
            def __init__(self, dims):
                self.dim = dims
    
        class MockTensorType:
            def __init__(self, elem_type, shape=None):
                self.elem_type = elem_type
                self.shape = shape
    
            def HasField(self, field):
                if field == "shape":
                    return self.shape is not None
                return False
    
        class MockTypeProto:
            def __init__(self, tensor_type):
                self.tensor_type = tensor_type
    
            def HasField(self, field):
                return field == "tensor_type"
    
        # Test cases
        tensor_type = MockTensorType(onnx.TensorProto.FLOAT, MockShape([MockDim(10), MockDim(dim_param='N')]))
        type_proto = MockTypeProto(tensor_type)
        self.assertEqual(onnx_type_to_onnxscript_repr(type_proto), "FLOAT[10,'N']")
    
        tensor_type = MockTensorType(onnx.TensorProto.INT32, MockShape([]))
        type_proto = MockTypeProto(tensor_type)
        self.assertEqual(onnx_type_to_onnxscript_repr(type_proto), "INT32")
    
        tensor_type = MockTensorType(onnx.TensorProto.BOOL)
        type_proto = MockTypeProto(tensor_type)
        self.assertEqual(onnx_type_to_onnxscript_repr(type_proto), "BOOL[...]")


    def test_check_dim_invalid_type(self):
        with self.assertRaises(TypeError):
            _check_dim(3.14)
        with self.assertRaises(TypeError):
            _check_dim([1, 2, 3])
        with self.assertRaises(TypeError):
            _check_dim({'dim': 1})


if __name__ == "__main__":
    unittest.main()
