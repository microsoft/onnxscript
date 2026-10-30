# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx_ir as ir

import onnxscript
from onnxscript import onnx_types
from onnxscript._internal import type_annotation
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tensor import Tensor


class TensorTypeCoverageTest(unittest.TestCase):
    def test_all_ir_dtypes_have_public_annotations(self):
        for dtype in ir.DataType:
            if dtype == ir.DataType.UNDEFINED:
                continue
            with self.subTest(dtype=dtype):
                tensor_type = onnx_types.tensor_type_registry[dtype]
                self.assertIs(getattr(onnxscript, dtype.name), tensor_type)
                self.assertIn(dtype.name, onnxscript.__all__)
                self.assertEqual(tensor_type.to_string(), f"tensor({dtype.name.lower()})")
                self.assertEqual(
                    tensor_type[2, "N"].to_ir_type_and_shape().type, ir.TensorType(dtype)
                )
                proto = tensor_type[2, "N"].to_type_proto()
                self.assertEqual(proto.tensor_type.elem_type, dtype)
                self.assertEqual(
                    onnx_types.onnx_type_to_onnxscript_repr(proto), f"{dtype.name}[2,'N']"
                )

    def test_schema_type_strings_match_installed_onnx(self):
        expected = {
            f"tensor({name.lower()})"
            for name, value in onnx.TensorProto.DataType.items()
            if value != onnx.TensorProto.UNDEFINED
        }
        self.assertEqual(set(type_annotation.ALL_TENSOR_TYPE_STRINGS), expected)

    def test_numeric_tensor_dtype_and_storage_roundtrip(self):
        for dtype in ir.DataType:
            if dtype in (ir.DataType.UNDEFINED, ir.DataType.STRING):
                continue
            with self.subTest(dtype=dtype):
                array = np.array([1, 1], dtype=dtype.numpy())
                self.assertEqual(Tensor(array).onnx_dtype, dtype)
                tensor = ir.tensor(array)
                restored = ir.from_proto(ir.to_proto(tensor))
                self.assertEqual(restored.dtype, dtype)
                np.testing.assert_array_equal(restored.numpy(), array)


class TensorTypeToIrTest(unittest.TestCase):
    """Tests for TensorType.to_ir_type_and_shape()."""

    def test_scalar_type(self):
        """FLOAT (no subscript) maps to rank-0 tensor (empty shape)."""
        ts = FLOAT.to_ir_type_and_shape()
        self.assertIsInstance(ts, ir.TypeAndShape)
        self.assertEqual(ts.type, ir.TensorType(ir.DataType.FLOAT))
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 0)

    def test_unknown_rank(self):
        """FLOAT[...] maps to unknown-rank (shape=None)."""
        ts = FLOAT[...].to_ir_type_and_shape()
        self.assertIsInstance(ts, ir.TypeAndShape)
        self.assertIsNone(ts.shape)

    def test_single_dim(self):
        """FLOAT[1024] maps to a 1-D tensor with dimension 1024."""
        ts = FLOAT[1024].to_ir_type_and_shape()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 1)
        self.assertEqual(ts.shape[0], 1024)

    def test_multi_dim_int(self):
        """FLOAT[3, 4] maps to a 2-D tensor with dims (3, 4)."""
        ts = FLOAT[3, 4].to_ir_type_and_shape()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 2)
        self.assertEqual(ts.shape[0], 3)
        self.assertEqual(ts.shape[1], 4)

    def test_symbolic_dims(self):
        """FLOAT['M', 'N'] maps to a 2-D tensor with symbolic dims."""
        ts = FLOAT["M", "N"].to_ir_type_and_shape()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 2)

    def test_other_dtype(self):
        """INT64[...] preserves the correct dtype."""
        ts = INT64[...].to_ir_type_and_shape()
        self.assertEqual(ts.type, ir.TensorType(ir.DataType.INT64))


if __name__ == "__main__":
    unittest.main()
