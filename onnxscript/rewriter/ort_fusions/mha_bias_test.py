# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for FuseBiasMHA rule (mha_bias.py).

The rule detects Add(matmul, bias) before MultiHeadAttention inputs and
fuses the biases into the MHA's consolidated bias input (Concat of q/k/v biases).
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript import FLOAT, INT32, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions import _test_utils as test_utils
from onnxscript.rewriter.ort_fusions.mha_bias import fuse_mha_bias

msft_op = values.Opset("com.microsoft", 1)

_B, _S, _D, _Dk, _Dv = 2, 8, 16, 16, 16
_NUM_HEADS = 4


def _count_op(model: ir.Model, op_type: str, domain: str = "") -> int:
    return sum(1 for node in model.graph if node.op_type == op_type and node.domain == domain)


# --- Script models for positive tests ---


@script()
def _mha_all_biases(query_matmul, key_matmul, value_matmul, q_bias, k_bias, v_bias):
    query = op.Add(query_matmul, q_bias)
    key = op.Add(key_matmul, k_bias)
    value = op.Add(value_matmul, v_bias)
    return msft_op.MultiHeadAttention(query, key, value, num_heads=_NUM_HEADS)


@script()
def _mha_q_bias_only(query_matmul, key_matmul, value_matmul, q_bias):
    query = op.Add(query_matmul, q_bias)
    return msft_op.MultiHeadAttention(query, key_matmul, value_matmul, num_heads=_NUM_HEADS)


@script()
def _mha_k_bias_only(query_matmul, key_matmul, value_matmul, k_bias):
    key = op.Add(key_matmul, k_bias)
    return msft_op.MultiHeadAttention(query_matmul, key, value_matmul, num_heads=_NUM_HEADS)


@script()
def _mha_v_bias_only(query_matmul, key_matmul, value_matmul, v_bias):
    value = op.Add(value_matmul, v_bias)
    return msft_op.MultiHeadAttention(query_matmul, key_matmul, value, num_heads=_NUM_HEADS)


@script()
def _mha_qk_biases(query_matmul, key_matmul, value_matmul, q_bias, k_bias):
    query = op.Add(query_matmul, q_bias)
    key = op.Add(key_matmul, k_bias)
    return msft_op.MultiHeadAttention(query, key, value_matmul, num_heads=_NUM_HEADS)


# --- Script models for negative tests ---


@script()
def _mha_no_biases(query_matmul, key_matmul, value_matmul):
    return msft_op.MultiHeadAttention(
        query_matmul, key_matmul, value_matmul, num_heads=_NUM_HEADS
    )


@script()
def _mha_int32_with_bias(query_matmul, key_matmul, value_matmul, q_bias):
    query = op.Add(query_matmul, q_bias)
    return msft_op.MultiHeadAttention(query, key_matmul, value_matmul, num_heads=_NUM_HEADS)


@script()
def _mha_rank2_query_with_bias(query_matmul, key_matmul, value_matmul, q_bias):
    query = op.Add(query_matmul, q_bias)
    return msft_op.MultiHeadAttention(query, key_matmul, value_matmul, num_heads=_NUM_HEADS)


class FuseBiasMHATest(unittest.TestCase):
    """Unit tests for the FuseBiasMHA rewrite rule."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _apply(self, model: ir.Model) -> int:
        return fuse_mha_bias(model)

    def _check_numerical_equivalence(self, model: ir.Model, inputs: dict, expected_count: int):
        """Apply fusion and verify the result is numerically equivalent."""
        original_output = test_utils.ort_run("Original", model, inputs)
        count = self._apply(model)
        self.assertEqual(count, expected_count)
        fused_output = test_utils.ort_run("Fused", model, inputs)
        test_utils.assert_allclose(original_output, fused_output)

    # --- Positive tests: fusion should apply with numerical validation ---

    def test_all_three_biases_fused(self):
        """Q, K, V all have bias Adds → fused into MHA bias input."""
        model = self._build(
            _mha_all_biases,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_D],
                FLOAT[_Dk],
                FLOAT[_Dv],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "query_matmul": np.random.randn(_B, _S, _D).astype(np.float32),
            "key_matmul": np.random.randn(_B, _S, _Dk).astype(np.float32),
            "value_matmul": np.random.randn(_B, _S, _Dv).astype(np.float32),
            "q_bias": np.random.randn(_D).astype(np.float32),
            "k_bias": np.random.randn(_Dk).astype(np.float32),
            "v_bias": np.random.randn(_Dv).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(_count_op(model, "Add"), 0)
        self.assertEqual(_count_op(model, "Concat"), 1)
        self.assertEqual(_count_op(model, "MultiHeadAttention", "com.microsoft"), 1)

    def test_only_q_bias(self):
        """Only query has a bias Add → still fuses (k/v get zero biases)."""
        model = self._build(
            _mha_q_bias_only,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_D],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "query_matmul": np.random.randn(_B, _S, _D).astype(np.float32),
            "key_matmul": np.random.randn(_B, _S, _Dk).astype(np.float32),
            "value_matmul": np.random.randn(_B, _S, _Dv).astype(np.float32),
            "q_bias": np.random.randn(_D).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(_count_op(model, "Add"), 0)
        self.assertEqual(_count_op(model, "Concat"), 1)

    def test_only_k_bias(self):
        """Only key has a bias Add → still fuses."""
        model = self._build(
            _mha_k_bias_only,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_Dk],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "query_matmul": np.random.randn(_B, _S, _D).astype(np.float32),
            "key_matmul": np.random.randn(_B, _S, _Dk).astype(np.float32),
            "value_matmul": np.random.randn(_B, _S, _Dv).astype(np.float32),
            "k_bias": np.random.randn(_Dk).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(_count_op(model, "Add"), 0)

    def test_only_v_bias(self):
        """Only value has a bias Add → still fuses."""
        model = self._build(
            _mha_v_bias_only,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_Dv],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "query_matmul": np.random.randn(_B, _S, _D).astype(np.float32),
            "key_matmul": np.random.randn(_B, _S, _Dk).astype(np.float32),
            "value_matmul": np.random.randn(_B, _S, _Dv).astype(np.float32),
            "v_bias": np.random.randn(_Dv).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)
        self.assertEqual(_count_op(model, "Add"), 0)

    def test_q_and_k_bias_only(self):
        """Q and K have biases, V does not → still fuses."""
        model = self._build(
            _mha_qk_biases,
            input_types=[
                FLOAT["B", "S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_D],
                FLOAT[_Dk],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        inputs = {
            "query_matmul": np.random.randn(_B, _S, _D).astype(np.float32),
            "key_matmul": np.random.randn(_B, _S, _Dk).astype(np.float32),
            "value_matmul": np.random.randn(_B, _S, _Dv).astype(np.float32),
            "q_bias": np.random.randn(_D).astype(np.float32),
            "k_bias": np.random.randn(_Dk).astype(np.float32),
        }
        self._check_numerical_equivalence(model, inputs, expected_count=1)

    # --- Negative tests: fusion should NOT apply ---

    def test_no_biases_no_fusion(self):
        """No bias Adds at all → rule should not apply."""
        model = self._build(
            _mha_no_biases,
            input_types=[FLOAT["B", "S", _D], FLOAT["B", "S", _Dk], FLOAT["B", "S", _Dv]],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)
        self.assertEqual(_count_op(model, "Concat"), 0)

    def test_int32_dtype_no_fusion(self):
        """Integer query dtype → check rejects non-float types."""
        model = self._build(
            _mha_int32_with_bias,
            input_types=[
                INT32["B", "S", _D],
                INT32["B", "S", _Dk],
                INT32["B", "S", _Dv],
                INT32[_D],
            ],
            output_types=[INT32["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)

    def test_shape_mismatch_no_fusion(self):
        """Query with rank-2 shape [S, D] instead of [B, S, D] → check rejects."""
        model = self._build(
            _mha_rank2_query_with_bias,
            input_types=[
                FLOAT["S", _D],
                FLOAT["B", "S", _Dk],
                FLOAT["B", "S", _Dv],
                FLOAT[_D],
            ],
            output_types=[FLOAT["B", "S", _D]],
        )
        count = self._apply(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
