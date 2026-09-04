# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for RmsNormFusion rules (rms_normalization.py).

The rule detects the RMS-normalization pattern:
    x_norm = x / sqrt(mean(x^2) + eps)
    output = x_norm * scale
and fuses it into SimplifiedLayerNormalization.

Covers both mul-orderings, optional Casts (mixed-precision),
and negative cases (bad dtype, non-scalar epsilon).
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
from parameterized import parameterized

from onnxscript import FLOAT, FLOAT16, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions import _test_utils as test_utils
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization

_B, _S, _D = 2, 8, 16
_EPS = ir.tensor(np.array([1e-6], dtype=np.float32))


# --- Pattern: Mul(scale, normalized) — mul_order=False ---


@script()
def _rms_norm_scale_first(x, scale):
    x_sq = op.Pow(x, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x, inv_rms)
    return op.Mul(scale, normalized)


# --- Pattern: Mul(normalized, scale) — mul_order=True ---


@script()
def _rms_norm_norm_first(x, scale):
    x_sq = op.Pow(x, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x, inv_rms)
    return op.Mul(normalized, scale)


# --- Pattern with Cast on input (mixed-precision: fp16 input, fp32 compute) ---


@script()
def _rms_norm_with_cast_input(x, scale):
    x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
    x_sq = op.Pow(x_f32, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_f32, inv_rms)
    result = op.Cast(normalized, to=ir.DataType.FLOAT16)
    return op.Mul(result, scale)


# --- Negative: integer input ---


@script()
def _rms_norm_int_input(x, scale):
    x_f = op.Cast(x, to=ir.DataType.FLOAT)
    x_sq = op.Pow(x_f, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_f, inv_rms)
    return op.Mul(normalized, scale)


class RmsNormFusionTest(unittest.TestCase):
    """Unit tests for RmsNormFusion rewrite rules."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _apply(self, model: ir.Model) -> int:
        return fuse_rms_normalization(model)

    def _count_op(self, model: ir.Model, op_type: str) -> int:
        return sum(1 for n in model.graph if n.op_type == op_type)

    def _check_numerical_equivalence(self, model: ir.Model, inputs: dict, expected_count: int):
        original_output = test_utils.ort_run("Original", model, inputs)
        count = self._apply(model)
        self.assertEqual(count, expected_count)
        fused_output = test_utils.ort_run("Fused", model, inputs)
        test_utils.assert_allclose(original_output, fused_output)

    # --- Positive tests ---

    @parameterized.expand(
        [
            ("scale_times_normalized", _rms_norm_scale_first),
            ("normalized_times_scale", _rms_norm_norm_first),
        ]
    )
    def test_mul_order_variants(self, _name, script_fn):
        """Both Mul orderings (scale*norm and norm*scale) should fuse."""
        model = self._build(
            script_fn,
            input_types=[FLOAT["B", "S", _D], FLOAT[_D]],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "x": np.random.randn(_B, _S, _D).astype(np.float32),
            "scale": np.random.randn(_D).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(self._count_op(model, "SimplifiedLayerNormalization"), 1)
        self.assertEqual(self._count_op(model, "Pow"), 0)
        self.assertEqual(self._count_op(model, "ReduceMean"), 0)

    def test_cast_input_mixed_precision(self):
        """fp16 input Cast to fp32 for compute, Cast back → still fuses."""
        model = self._build(
            _rms_norm_with_cast_input,
            input_types=[FLOAT16["B", "S", _D], FLOAT16[_D]],
            output_types=[FLOAT16["B", "S", _D]],
        )
        inputs = {
            "x": np.random.randn(_B, _S, _D).astype(np.float16),
            "scale": np.random.randn(_D).astype(np.float16),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(self._count_op(model, "SimplifiedLayerNormalization"), 1)

    # --- Negative tests ---

    def test_int_input_no_fusion(self):
        """Integer input dtype → check rejects (x.dtype not in float_types)."""
        from onnxscript import INT32

        model = self._build(
            _rms_norm_int_input,
            input_types=[INT32["B", "S", _D], FLOAT[_D]],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)
        self.assertEqual(self._count_op(model, "SimplifiedLayerNormalization"), 0)


if __name__ == "__main__":
    unittest.main()
