# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for FuseBiasMHA rule (mha_bias.py).

The rule detects Add(matmul, bias) before MultiHeadAttention inputs and
fuses matching 1D biases into the MHA's consolidated bias input (Concat of q/k/v biases).
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import parameterized

from onnxscript import FLOAT, INT32, script, values
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions import _test_utils as test_utils
from onnxscript.rewriter.ort_fusions import optimize_for_ort
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


@script()
def _attention_with_position(query_matmul, key_matmul, value_matmul, q_bias):
    query = op.Add(query_matmul, q_bias)
    shape = op.Constant(value_ints=[0, 0, _NUM_HEADS, -1])
    query = op.Transpose(op.Reshape(query, shape), perm=[0, 2, 1, 3])
    key = op.Transpose(op.Reshape(key_matmul, shape), perm=[0, 2, 3, 1])
    value = op.Transpose(op.Reshape(value_matmul, shape), perm=[0, 2, 1, 3])
    scores = op.Mul(op.MatMul(query, key), 0.5)
    output = op.MatMul(op.Softmax(scores, axis=-1), value)
    return op.Reshape(op.Transpose(output, perm=[0, 2, 1, 3]), [0, 0, -1])


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

    @parameterized.parameterized.expand(
        [
            ("query", _mha_q_bias_only, "q_bias"),
            ("key", _mha_k_bias_only, "k_bias"),
            ("value", _mha_v_bias_only, "v_bias"),
        ]
    )
    def test_position_embedding_not_packed(self, _, script_fn, bias_name):
        for bias_shape in [(_B, _S, _D), (1, _D), (1,)]:
            with self.subTest(bias_shape=bias_shape):
                model = self._build(
                    script_fn,
                    input_types=[FLOAT[_B, _S, _D]] * 3 + [FLOAT[bias_shape]],
                    output_types=[FLOAT[_B, _S, _D]],
                )
                rng = np.random.default_rng(0)
                inputs = {
                    name: rng.standard_normal((_B, _S, _D)).astype(np.float32)
                    for name in ("query_matmul", "key_matmul", "value_matmul")
                }
                inputs[bias_name] = rng.standard_normal(bias_shape).astype(np.float32)
                self._check_numerical_equivalence(model, inputs, expected_count=0)

    @parameterized.parameterized.expand(
        [("position_embedding", FLOAT["B", "S", _D], 0), ("bias", FLOAT[_D], 1)]
    )
    def test_bias_shape_full_pipeline(self, _, bias_type, expected_count):
        model = self._build(
            _attention_with_position,
            input_types=[FLOAT["B", "S", _D]] * 3 + [bias_type],
            output_types=[FLOAT["B", "S", _D]],
        )
        rng = np.random.default_rng(0)
        inputs = {
            name: rng.standard_normal((_B, _S, _D)).astype(np.float32)
            for name in ("query_matmul", "key_matmul", "value_matmul")
        }
        bias_shape = (_D,) if expected_count else (_B, _S, _D)
        inputs["q_bias"] = rng.standard_normal(bias_shape).astype(np.float32)
        original_outputs = test_utils.ort_run("original", model, inputs)
        model, counts = optimize_for_ort(model)
        self.assertGreater(counts["mha1"] + counts["mha2"], 0)
        self.assertEqual(counts["mha_bias"], expected_count)
        self.assertEqual(_count_op(model, "Add"), 1 - expected_count)
        test_utils.assert_allclose(
            test_utils.ort_run("optimized", model, inputs), original_outputs
        )

    @parameterized.parameterized.expand(
        [("unknown", None), ("unknown_length", ["D"]), ("scalar", [])]
    )
    def test_unproven_bias_shape_not_packed(self, _, shape):
        model = self._build(
            _mha_q_bias_only,
            input_types=[FLOAT["B", "S", _D]] * 3 + [FLOAT[_D]],
            output_types=[FLOAT["B", "S", _D]],
        )
        model.graph.inputs[3].shape = ir.Shape(shape) if shape is not None else None
        self.assertEqual(self._apply(model), 0)

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
