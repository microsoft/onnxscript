# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended unit tests for RmsNormFusion rules (rules/fusion/_rms_normalization.py).

Adds coverage for: mul_order variants, mixed-precision Cast paths,
double dtype, and negative cases (int input, non-float compute dtype).
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
from parameterized import parameterized

from onnxscript import DOUBLE, FLOAT, FLOAT16, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter import testing as rewriter_testing
from onnxscript.rewriter.rules.fusion._rms_normalization import fuse_rms_normalization

_EPS = ir.tensor(np.array([1e-6], dtype=np.float32))
_EPS_D = ir.tensor(np.array([1e-6], dtype=np.float64))


# --- mul_order=False: Mul(scale, normalized) ---


@script()
def _rms_scale_first(x, scale):
    x_sq = op.Pow(x, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x, inv_rms)
    return op.Mul(scale, normalized)


# --- mul_order=True: Mul(normalized, scale) ---


@script()
def _rms_norm_first(x, scale):
    x_sq = op.Pow(x, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x, inv_rms)
    return op.Mul(normalized, scale)


# --- Mixed-precision: fp16 input, fp32 compute, fp16 output ---


@script()
def _rms_mixed_precision(x, scale):
    x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
    x_sq = op.Pow(x_f32, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_f32, inv_rms)
    result = op.Cast(normalized, to=ir.DataType.FLOAT16)
    return op.Mul(result, scale)


# --- Double precision ---


@script()
def _rms_double(x, scale):
    x_sq = op.Pow(x, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS_D)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x, inv_rms)
    return op.Mul(normalized, scale)


# --- Negative: int32 input ---


@script()
def _rms_int_input(x, scale):
    x_f = op.Cast(x, to=ir.DataType.FLOAT)
    x_sq = op.Pow(x_f, 2.0)
    mean_sq = op.ReduceMean(x_sq, [-1], keepdims=1, noop_with_empty_axes=0)
    eps = op.Constant(value=_EPS)
    rms = op.Sqrt(op.Add(mean_sq, eps))
    inv_rms = op.Reciprocal(rms)
    normalized = op.Mul(x_f, inv_rms)
    return op.Mul(normalized, scale)


class RmsNormOnnxFusionExtendedTest(unittest.TestCase):
    """Extended tests for RmsNormFusion (rules/fusion variant producing RMSNormalization)."""

    _B, _S, _D = 2, 4, 16

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _count_op(self, model: ir.Model, op_type: str) -> int:
        return sum(1 for n in model.graph if n.op_type == op_type)

    def _check_numerical_equivalence(self, model: ir.Model, inputs: dict, expected_count: int):
        """Apply fusion and verify numerical equivalence using ONNX reference impl.

        ORT does not yet have a kernel for RMSNormalization, so we use
        the ONNX reference implementation for validation.
        """
        original_proto = ir.serde.serialize_model(model)
        count = fuse_rms_normalization(model)
        self.assertEqual(count, expected_count)
        fused_proto = ir.serde.serialize_model(model)
        rewriter_testing.assert_numerically_equal(
            original_proto, fused_proto, args=inputs, use_reference=True
        )

    # --- Positive tests ---

    @parameterized.expand(
        [
            ("scale_times_normalized", _rms_scale_first),
            ("normalized_times_scale", _rms_norm_first),
        ]
    )
    def test_mul_order_variants(self, _name, script_fn):
        """Both Mul orderings should fuse to RMSNormalization."""
        model = self._build(
            script_fn,
            input_types=[FLOAT[self._B, self._S, self._D], FLOAT[self._D]],
            output_types=[FLOAT[self._B, self._S, self._D]],
        )
        inputs = {
            "x": np.random.randn(self._B, self._S, self._D).astype(np.float32),
            "scale": np.random.randn(self._D).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(self._count_op(model, "RMSNormalization"), 1)
        self.assertEqual(self._count_op(model, "Pow"), 0)

    def test_mixed_precision_cast(self):
        """fp16 input Cast to fp32 for compute, Cast back → fuses."""
        model = self._build(
            _rms_mixed_precision,
            input_types=[FLOAT16[self._B, self._S, self._D], FLOAT16[self._D]],
            output_types=[FLOAT16[self._B, self._S, self._D]],
        )
        inputs = {
            "x": np.random.randn(self._B, self._S, self._D).astype(np.float16),
            "scale": np.random.randn(self._D).astype(np.float16),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(self._count_op(model, "RMSNormalization"), 1)

    def test_double_precision(self):
        """Double-precision inputs → fuses (double is a valid compute type).

        Structural check only: ONNX reference impl does not support
        RMSNormalization with stash_type=DOUBLE.
        """
        model = self._build(
            _rms_double,
            input_types=[DOUBLE[self._B, self._S, self._D], DOUBLE[self._D]],
            output_types=[DOUBLE[self._B, self._S, self._D]],
        )
        count = fuse_rms_normalization(model)
        self.assertEqual(count, 1)
        self.assertEqual(self._count_op(model, "RMSNormalization"), 1)

    # --- Negative tests ---

    def test_int_input_no_fusion(self):
        """Integer input dtype → check rejects (x.dtype not in float_types)."""
        from onnxscript import INT32

        model = self._build(
            _rms_int_input,
            input_types=[INT32["B", "S", 16], FLOAT[16]],
            output_types=[FLOAT["B", "S", 16]],
        )
        count = fuse_rms_normalization(model)
        self.assertEqual(count, 0)
        self.assertEqual(self._count_op(model, "RMSNormalization"), 0)


if __name__ == "__main__":
    unittest.main()
