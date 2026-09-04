# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended unit tests for LayerNormFusion and LayerNormBiasFusion rules.

Adds coverage for: OrValue alternatives (Pow vs Mul for deviation²,
Div vs Mul+Reciprocal for normalization), double precision, and negative cases.
"""

from __future__ import annotations

import unittest

import onnx_ir as ir

import onnxscript.optimizer
import onnxscript.rewriter.testing
from onnxscript import DOUBLE, FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter.rules.fusion._layer_norm import fuse_layer_normalization

# --- Pow variant for deviation_squared ---


@script()
def _ln_pow_variant(x: FLOAT[2, 4, 8], scale: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """Uses Pow(deviation, 2) instead of Mul(deviation, deviation)."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Pow(deviation, 2)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    inv_std_dev = op.Reciprocal(std_dev)
    normalized = op.Mul(deviation, inv_std_dev)
    return op.Mul(normalized, scale)


# --- Div variant for normalization ---


@script()
def _ln_div_variant(x: FLOAT[2, 4, 8], scale: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """Uses Div(deviation, std_dev) instead of Mul(deviation, Reciprocal(std_dev))."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Mul(deviation, deviation)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    normalized = op.Div(deviation, std_dev)
    return op.Mul(normalized, scale)


# --- Pow + Div combined (both OrValue alternatives) ---


@script()
def _ln_pow_div(x: FLOAT[2, 4, 8], scale: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """Both alternative branches: Pow for deviation², Div for normalization."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Pow(deviation, 2)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    normalized = op.Div(deviation, std_dev)
    return op.Mul(normalized, scale)


# --- Div variant with bias ---


@script()
def _ln_div_with_bias(x: FLOAT[2, 4, 8], scale: FLOAT[8], bias: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """Div normalization path + bias addition."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Mul(deviation, deviation)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    normalized = op.Div(deviation, std_dev)
    scaled = op.Mul(normalized, scale)
    return op.Add(scaled, bias)


# --- Double precision ---


@script()
def _ln_double(x: DOUBLE[2, 4, 8], scale: DOUBLE[8]) -> DOUBLE[2, 4, 8]:
    """Double-precision inputs."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Mul(deviation, deviation)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    inv_std_dev = op.Reciprocal(std_dev)
    normalized = op.Mul(deviation, inv_std_dev)
    return op.Mul(normalized, scale)


# --- Negative: float16 input (not in LAYER_NORM_COMPUTE_TYPES) ---


@script()
def _ln_fp16(x: FLOAT[2, 4, 8], scale: FLOAT[8]) -> FLOAT[2, 4, 8]:
    """Pattern is structurally correct but input dtype will be fp16."""
    mean = op.ReduceMean(x, [-1], keepdims=1)
    deviation = op.Sub(x, mean)
    deviation_squared = op.Mul(deviation, deviation)
    variance = op.ReduceMean(deviation_squared, [-1], keepdims=1)
    epsilon = op.Constant(value_float=1e-5)
    std_dev = op.Sqrt(op.Add(variance, epsilon))
    inv_std_dev = op.Reciprocal(std_dev)
    normalized = op.Mul(deviation, inv_std_dev)
    return op.Mul(normalized, scale)


class LayerNormFusionExtendedTest(unittest.TestCase):
    """Extended tests for LayerNormFusion (OrValue alternatives, dtypes, negatives)."""

    def _check(self, test_script, expected_op="LayerNormalization"):
        """Build, fuse, verify single fused node, numerical equivalence."""
        model_proto = test_script.to_model_proto()
        input_data = onnxscript.rewriter.testing.generate_random_inputs(model_proto)
        model = ir.serde.deserialize_model(model_proto)
        count = fuse_layer_normalization(model)
        self.assertGreater(count, 0)
        onnxscript.optimizer.remove_unused_nodes(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn(expected_op, op_types)
        fused_proto = ir.serde.serialize_model(model)
        onnxscript.rewriter.testing.assert_numerically_equal(
            model_proto, fused_proto, input_data
        )

    # --- Positive tests: OrValue alternatives ---

    def test_pow_for_deviation_squared(self):
        """Pow(deviation, 2) instead of Mul(deviation, deviation) → fuses."""
        self._check(_ln_pow_variant)

    def test_div_for_normalization(self):
        """Div(deviation, std_dev) instead of Mul(deviation, Reciprocal(std_dev)) → fuses."""
        self._check(_ln_div_variant)

    def test_pow_and_div_combined(self):
        """Both OrValue alternative branches active → fuses."""
        self._check(_ln_pow_div)

    def test_div_with_bias(self):
        """Div normalization + bias → fuses into LayerNormalization with bias."""
        self._check(_ln_div_with_bias)

    def test_double_precision(self):
        """Double-precision inputs → fuses (double is a valid compute type)."""
        model_proto = _ln_double.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        count = fuse_layer_normalization(model)
        self.assertGreater(count, 0)
        onnxscript.optimizer.remove_unused_nodes(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("LayerNormalization", op_types)

    # --- Negative test ---

    def test_fp16_input_no_fusion(self):
        """float16 input dtype → check rejects (not in LAYER_NORM_COMPUTE_TYPES)."""
        from onnxscript import FLOAT16

        model_proto = _ln_fp16.to_model_proto(
            input_types=[FLOAT16[2, 4, 8], FLOAT16[8]],
            output_types=[FLOAT16[2, 4, 8]],
        )
        model = ir.serde.deserialize_model(model_proto)
        count = fuse_layer_normalization(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
