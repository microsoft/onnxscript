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

# Pre-computed constant for use inside @script functions
_SCALE_VALUE = 0.25


# --- Script models ---


@script()
def _mha_with_scalar_scale(query, key, value, scale):
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

    def _make_scale_constant(self, model: ir.Model, scale_value: float):
        """Convert the ``scale`` graph input into a constant initializer."""
        for node in model.graph:
            if node.op_type == "Mul":
                scale_input = node.inputs[1]
                assert scale_input is not None
                scale_input.const_value = ir.tensor(np.array([scale_value], dtype=np.float32))
                model.graph.inputs.pop()
                return
        raise RuntimeError("Mul node not found")

    def _check_numerical_equivalence(
        self, model: ir.Model, inputs: dict, scale_value: float, expected_count: int
    ):
        # Run original model *before* making scale constant (scale is a graph input)
        inputs_with_scale = {
            **inputs,
            "scale": np.array([scale_value], dtype=np.float32),
        }
        original_output = test_utils.ort_run("Original", model, inputs_with_scale)
        # Now convert scale to constant and apply fusion
        self._make_scale_constant(model, scale_value)
        count = self._apply(model)
        self.assertEqual(count, expected_count)
        fused_output = test_utils.ort_run("Fused", model, inputs)
        test_utils.assert_allclose(original_output, fused_output)

    # --- Positive tests ---

    def _build_scale_model(self):
        return self._build(
            _mha_with_scalar_scale,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT[1],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )

    def _make_inputs(self):
        return {
            "query": np.random.randn(_B, _S, _D).astype(np.float32),
            "key": np.random.randn(_B, _S, _D).astype(np.float32),
            "value": np.random.randn(_B, _S, _D).astype(np.float32),
        }

    def test_scalar_scale_fused(self):
        """Mul(query, scalar_constant) before MHA → scale absorbed into attribute."""
        model = self._build_scale_model()
        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, _SCALE_VALUE, expected_count=1)
        # Verify Mul is gone and MHA has scale attribute
        self.assertFalse(any(n.op_type == "Mul" for n in model.graph), "Mul should be removed")
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = _SCALE_VALUE * _DEFAULT_SCALE
        self.assertAlmostEqual(scale_attr, expected, places=5)

    def test_integer_scale_fused(self):
        """Integer scale constant (e.g. 2) → still fused."""
        model = self._build_scale_model()
        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, 2.0, expected_count=1)
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = 2.0 * _DEFAULT_SCALE
        self.assertAlmostEqual(scale_attr, expected, places=5)

    def test_scale_combined_with_existing_scale_attr(self):
        """MHA already has a scale attribute → external scale is multiplied with it."""
        model = self._build_scale_model()
        # Set existing MHA scale attribute before any ORT run
        existing_scale = 0.1
        for node in model.graph:
            if node.op_type == "MultiHeadAttention" and node.domain == "com.microsoft":
                node.attributes["scale"] = ir.AttrFloat32("scale", existing_scale)

        inputs = self._make_inputs()
        self._check_numerical_equivalence(model, inputs, _SCALE_VALUE, expected_count=1)
        mha_node = self._get_mha_node(model)
        self.assertIsNotNone(mha_node)
        scale_attr = mha_node.attributes.get_float("scale", None)
        self.assertIsNotNone(scale_attr)
        expected = _SCALE_VALUE * existing_scale
        self.assertAlmostEqual(scale_attr, expected, places=5)

    # --- Negative tests ---

    def test_no_mul_no_fusion(self):
        """No Mul before MHA → rule does not match."""
        model = self._build(
            _mha_no_scale,
            input_types=[FLOAT["B", "S", _D], FLOAT["B", "S", _D], FLOAT["B", "S", _D]],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)

    def test_dynamic_scale_no_fusion(self):
        """Scale is a non-constant graph input → check rejects."""
        model = self._build(
            _mha_with_dynamic_scale,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _D],
                FLOAT[1],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
