# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for FuseMHAScale rule (mha_scale.py).

The rule detects Mul(query, constant_scale) before MultiHeadAttention and
fuses the scaling into the MHA's ``scale`` attribute.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import onnx_ir as ir

import onnxscript
import onnxscript.rewriter.ort_fusions._test_utils as test_utils
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions.mha_scale import fuse_mha_scale

msft_op = onnxscript.values.Opset("com.microsoft", 1)

_B, _S, _D = 2, 8, 16
_NUM_HEADS = 4
_HEAD_SIZE = _D // _NUM_HEADS
_DEFAULT_SCALE = 1.0 / math.sqrt(_HEAD_SIZE)

# Constants for use inside @script functions
_SCALE_TENSOR_025 = ir.tensor(np.array([0.25], dtype=np.float32))
_SCALE_TENSOR_2 = ir.tensor(np.array([2.0], dtype=np.float32))


# --- Script models ---


@script()
def _mha_with_scale_025(query, key, value):
    scale = op.Constant(value=_SCALE_TENSOR_025)
    scaled_q = op.Mul(query, scale)
    return msft_op.MultiHeadAttention(scaled_q, key, value, num_heads=_NUM_HEADS)


@script()
def _mha_with_scale_2(query, key, value):
    scale = op.Constant(value=_SCALE_TENSOR_2)
    scaled_q = op.Mul(query, scale)
    return msft_op.MultiHeadAttention(scaled_q, key, value, num_heads=_NUM_HEADS)


@script()
def _mha_no_scale(query, key, value):
    return msft_op.MultiHeadAttention(query, key, value, num_heads=_NUM_HEADS)


@script()
def _mha_with_dynamic_scale(query, key, value, scale):
    """Scale is a graph input (not constant) → fusion should not apply."""
    scaled_q = op.Mul(query, scale)
    return msft_op.MultiHeadAttention(scaled_q, key, value, num_heads=_NUM_HEADS)


class FuseMHAScaleTest(unittest.TestCase):
    """Unit tests for the FuseMHAScale rewrite rule."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _apply(self, model: ir.Model) -> int:
        return fuse_mha_scale(model)

    def _get_mha_node(self, model: ir.Model) -> ir.Node | None:
        for node in model.graph:
            if node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft":
                return node
        return None

    _3D = (FLOAT["B", "S", _D],) * 3
    _OUT = (FLOAT["B", "S", _D],)

    def _check_numerical_equivalence(self, model: ir.Model, inputs: dict, expected_count: int):
        original_output = test_utils.ort_run("Original", model, inputs)
        count = self._apply(model)
        self.assertEqual(count, expected_count)
        fused_output = test_utils.ort_run("Fused", model, inputs)
        test_utils.assert_allclose(original_output, fused_output)

    def _make_inputs(self):
        return {
            "query": np.random.randn(_B, _S, _D).astype(np.float32),
            "key": np.random.randn(_B, _S, _D).astype(np.float32),
            "value": np.random.randn(_B, _S, _D).astype(np.float32),
        }

    # --- Positive tests ---

    def test_scalar_scale_fused(self):
        """Mul(query, scalar_constant) before MHA → scale absorbed into attribute."""
        model = self._build(_mha_with_scale_025, self._3D, self._OUT)
        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertFalse(any(n.op_type == "Mul" for n in model.graph), "Mul should be removed")
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = 0.25 * _DEFAULT_SCALE
        self.assertAlmostEqual(scale_attr, expected, places=5)

    def test_integer_scale_fused(self):
        """Scale constant of 2.0 → still fused."""
        model = self._build(_mha_with_scale_2, self._3D, self._OUT)
        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = 2.0 * _DEFAULT_SCALE
        self.assertAlmostEqual(scale_attr, expected, places=5)

    def test_scale_combined_with_existing_scale_attr(self):
        """MHA already has a scale attribute → external scale is multiplied with it."""
        model = self._build(_mha_with_scale_025, self._3D, self._OUT)
        existing_scale = 0.1
        for node in model.graph:
            if node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft":
                node.attributes["scale"] = ir.AttrFloat32("scale", existing_scale)

        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = 0.25 * existing_scale
        self.assertAlmostEqual(scale_attr, expected, places=5)

    # --- Negative tests ---

    def test_no_mul_no_fusion(self):
        """No Mul before MHA → rule does not match."""
        model = self._build(_mha_no_scale, self._3D, self._OUT)
        count = self._apply(model)
        self.assertEqual(count, 0)

    def test_dynamic_scale_no_fusion(self):
        """Scale is a non-constant graph input → check rejects."""
        model = self._build(
            _mha_with_dynamic_scale,
            input_types=[*self._3D, FLOAT[1]],
            output_types=self._OUT,
        )
        count = self._apply(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
