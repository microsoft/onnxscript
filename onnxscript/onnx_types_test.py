# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest

import onnx_ir as ir

from onnxscript.onnx_types import FLOAT, INT64


class TensorTypeToIrTest(unittest.TestCase):
    """Tests for TensorType.to_ir()."""

    def test_scalar_type(self):
        """FLOAT (no subscript) maps to rank-0 tensor (empty shape)."""
        ts = FLOAT.to_ir()
        self.assertIsInstance(ts, ir.TypeAndShape)
        self.assertEqual(ts.type, ir.TensorType(ir.DataType.FLOAT))
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 0)

    def test_unknown_rank(self):
        """FLOAT[...] maps to unknown-rank (shape=None)."""
        ts = FLOAT[...].to_ir()
        self.assertIsInstance(ts, ir.TypeAndShape)
        self.assertIsNone(ts.shape)

    def test_single_dim(self):
        """FLOAT[1024] maps to a 1-D tensor with dimension 1024."""
        ts = FLOAT[1024].to_ir()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 1)
        self.assertEqual(ts.shape[0], 1024)

    def test_multi_dim_int(self):
        """FLOAT[3, 4] maps to a 2-D tensor with dims (3, 4)."""
        ts = FLOAT[3, 4].to_ir()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 2)
        self.assertEqual(ts.shape[0], 3)
        self.assertEqual(ts.shape[1], 4)

    def test_symbolic_dims(self):
        """FLOAT['M', 'N'] maps to a 2-D tensor with symbolic dims."""
        ts = FLOAT["M", "N"].to_ir()
        self.assertIsNotNone(ts.shape)
        self.assertEqual(len(ts.shape), 2)

    def test_other_dtype(self):
        """INT64[...] preserves the correct dtype."""
        ts = INT64[...].to_ir()
        self.assertEqual(ts.type, ir.TensorType(ir.DataType.INT64))


if __name__ == "__main__":
    unittest.main()
