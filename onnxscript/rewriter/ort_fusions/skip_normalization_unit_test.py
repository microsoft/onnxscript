# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for SkipRmsNormFusion and SkipLayerNormFusion rules.

SkipRmsNormFusion: Add(input, skip) → SimplifiedLayerNormalization →
    SkipSimplifiedLayerNormalization (com.microsoft).

SkipLayerNormFusion: Add(input, skip) → LayerNormalization →
    SkipLayerNormalization (com.microsoft).

Covers: no bias, post-add bias, pre-add bias variants; negative tests.
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
from onnxscript.rewriter.ort_fusions import optimize_for_ort
from onnxscript.rewriter.ort_fusions.skip_normalization import (
    fuse_skip_layer_normalization,
    fuse_skip_rms_normalization,
)

_B, _S, _D = 2, 8, 16


# ========== Skip RMS Norm patterns ==========


@script()
def _skip_rms_no_bias(input, skip, gamma):
    skip_sum = op.Add(input, skip)
    return op.SimplifiedLayerNormalization(
        skip_sum, gamma, axis=-1, epsilon=1e-6, stash_type=1
    )


@script()
def _skip_rms_no_bias_reversed(input, skip, gamma):
    """Skip + input order reversed (OrValue alternative)."""
    skip_sum = op.Add(skip, input)
    return op.SimplifiedLayerNormalization(
        skip_sum, gamma, axis=-1, epsilon=1e-6, stash_type=1
    )


@script()
def _skip_rms_post_bias(input, skip, gamma, bias):
    skip_sum = op.Add(input, skip)
    skip_sum_biased = op.Add(skip_sum, bias)
    return op.SimplifiedLayerNormalization(
        skip_sum_biased, gamma, axis=-1, epsilon=1e-6, stash_type=1
    )


@script()
def _skip_rms_pre_bias(input, skip, gamma, bias):
    input_biased = op.Add(input, bias)
    skip_sum = op.Add(input_biased, skip)
    return op.SimplifiedLayerNormalization(
        skip_sum, gamma, axis=-1, epsilon=1e-6, stash_type=1
    )


@script()
def _gemma_skip_rms(input, skip, gamma):
    residual = op.Add(input, skip)
    states = op.Cast(residual, to=ir.DataType.FLOAT)
    mean_square = op.ReduceMean(op.Pow(states, 2.0), [-1], keepdims=1, noop_with_empty_axes=0)
    normalized = states * op.Reciprocal(op.Sqrt(mean_square + 1e-6))
    return op.Cast(normalized * gamma, to=ir.DataType.FLOAT16), residual


# ========== Skip Layer Norm patterns ==========


@script()
def _skip_ln_no_bias(input, skip, gamma, beta):
    skip_sum = op.Add(input, skip)
    return op.LayerNormalization(skip_sum, gamma, beta, axis=-1, epsilon=1e-6, stash_type=1)


@script()
def _skip_ln_post_bias(input, skip, gamma, beta, bias):
    skip_sum = op.Add(input, skip)
    skip_sum_biased = op.Add(skip_sum, bias)
    return op.LayerNormalization(
        skip_sum_biased, gamma, beta, axis=-1, epsilon=1e-6, stash_type=1
    )


# ========== Negative patterns ==========


@script()
def _skip_rms_no_add(input, gamma):
    """No skip addition at all — just SimplifiedLayerNormalization."""
    return op.SimplifiedLayerNormalization(input, gamma, axis=-1, epsilon=1e-6, stash_type=1)


class SkipNormalizationTest(unittest.TestCase):
    """Unit tests for SkipRmsNormFusion and SkipLayerNormFusion."""

    def _build(self, script_fn, input_types, output_types) -> ir.Model:
        model_proto = script_fn.to_model_proto(
            input_types=input_types, output_types=output_types
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def _count_op(self, model: ir.Model, op_type: str, domain: str = "") -> int:
        return sum(1 for n in model.graph if n.op_type == op_type and n.domain == domain)

    def _check_numerical_equivalence(
        self, model: ir.Model, inputs: dict, fuse_fn, expected_count: int
    ):
        original_output = test_utils.ort_run("Original", model, inputs)
        count = fuse_fn(model)
        self.assertGreaterEqual(count, expected_count)
        fused_output = test_utils.ort_run("Fused", model, inputs)
        test_utils.assert_allclose(original_output, fused_output)

    _3D = FLOAT[_B, _S, _D]
    _1D = FLOAT[_D]

    def _make_inputs(self, *names):
        shapes = {
            "input": (_B, _S, _D),
            "skip": (_B, _S, _D),
            "gamma": (_D,),
            "beta": (_D,),
            "bias": (_D,),
        }
        return {n: np.random.randn(*shapes[n]).astype(np.float32) for n in names}

    # ---- Skip RMS Norm positive tests ----

    @parameterized.expand(
        [
            ("input_plus_skip", _skip_rms_no_bias),
            ("skip_plus_input", _skip_rms_no_bias_reversed),
        ]
    )
    def test_skip_rms_no_bias(self, _name, script_fn):
        """Skip + Input (both orderings) → SkipSimplifiedLayerNormalization."""
        model = self._build(
            script_fn,
            input_types=[self._3D, self._3D, self._1D],
            output_types=[self._3D],
        )
        inputs = self._make_inputs("input", "skip", "gamma")
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_rms_normalization, expected_count=1
        )
        self.assertEqual(
            self._count_op(model, "SkipSimplifiedLayerNormalization", "com.microsoft"), 1
        )
        self.assertEqual(self._count_op(model, "SimplifiedLayerNormalization"), 0)

    def test_skip_rms_post_bias(self):
        """(Input + Skip) + Bias → fuses with bias."""
        model = self._build(
            _skip_rms_post_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D],
            output_types=[self._3D],
        )
        inputs = self._make_inputs("input", "skip", "gamma", "bias")
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_rms_normalization, expected_count=1
        )
        self.assertEqual(
            self._count_op(model, "SkipSimplifiedLayerNormalization", "com.microsoft"), 1
        )

    def test_skip_rms_pre_bias(self):
        """(Input + Bias) + Skip → fuses with pre-add bias."""
        model = self._build(
            _skip_rms_pre_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D],
            output_types=[self._3D],
        )
        inputs = self._make_inputs("input", "skip", "gamma", "bias")
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_rms_normalization, expected_count=1
        )
        self.assertEqual(
            self._count_op(model, "SkipSimplifiedLayerNormalization", "com.microsoft"), 1
        )

    def _mixed_precision_rms(
        self, script_fn, scale, cast_dtype=ir.DataType.FLOAT16, keep_gamma_input=False
    ):
        has_bias = script_fn in (_skip_rms_post_bias, _skip_rms_pre_bias)
        input_types = [FLOAT16[_B, _S, _D], FLOAT16[_B, _S, _D], FLOAT[_D]]
        if has_bias:
            input_types.append(FLOAT16[_D])
        model = self._build(script_fn, input_types, [FLOAT[_B, _S, _D]])
        if scale is not None:
            gamma = model.graph.inputs[2]
            gamma.const_value = ir.tensor(scale)
            if not keep_gamma_input:
                model.graph.inputs.remove(gamma)
            model.graph.register_initializer(gamma)
        normalized = model.graph.outputs[0]
        cast = ir.Node(
            "",
            "Cast",
            [normalized],
            attributes=[ir.AttrInt64("to", cast_dtype)],
            outputs=[ir.Value(shape=normalized.shape, type=ir.TensorType(cast_dtype))],
        )
        model.graph.append(cast)
        model.graph.outputs[0] = cast.outputs[0]
        # Preserve the residual as a graph output, as in a transformer block.
        residual = normalized.producer().inputs[0]
        model.graph.outputs.append(residual)
        return model, normalized

    @parameterized.expand(
        [
            ("no_bias", _skip_rms_no_bias),
            ("reversed", _skip_rms_no_bias_reversed),
            ("post_bias", _skip_rms_post_bias),
            ("pre_bias", _skip_rms_pre_bias),
        ]
    )
    def test_skip_rms_exact_float32_scale(self, _name, script_fn):
        scale = np.linspace(0.5, 2.0, _D).astype(np.float16).astype(np.float32)
        model, _ = self._mixed_precision_rms(script_fn, scale)
        names = ["input", "skip"]
        if script_fn in (_skip_rms_post_bias, _skip_rms_pre_bias):
            names.append("bias")
        inputs = {k: v.astype(np.float16) for k, v in self._make_inputs(*names).items()}
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_rms_normalization, expected_count=1
        )
        fused = next(n for n in model.graph if n.op_type == "SkipSimplifiedLayerNormalization")
        self.assertTrue(all(v.dtype == ir.DataType.FLOAT16 for v in fused.inputs))
        np.testing.assert_array_equal(fused.inputs[2].const_value.numpy(), scale)
        self.assertEqual(self._count_op(model, "Cast"), 0)
        self.assertEqual(model.graph.outputs[1].dtype, ir.DataType.FLOAT16)

    def test_skip_rms_overridable_float32_scale(self):
        scale = np.ones((_D,), dtype=np.float32)
        model, _ = self._mixed_precision_rms(
            _skip_rms_no_bias, scale, keep_gamma_input=True
        )
        inputs = {
            k: v.astype(np.float16) for k, v in self._make_inputs("input", "skip").items()
        }
        inputs["gamma"] = np.linspace(0.5, 2.0, _D).astype(np.float32)
        original_output = test_utils.ort_run("Original", model, inputs)
        self.assertEqual(fuse_skip_rms_normalization(model), 0)
        test_utils.assert_allclose(
            test_utils.ort_run("Unfused", model, inputs), original_output
        )

    @parameterized.expand(
        [
            ("dynamic_scale", None, ir.DataType.FLOAT16, None),
            ("inexact_scale", 1.0001, ir.DataType.FLOAT16, None),
            ("overflow_scale", 1e10, ir.DataType.FLOAT16, None),
            ("no_cast_back", 1.0, None, None),
            ("wrong_cast_dtype", 1.0, ir.DataType.DOUBLE, None),
            ("float32_graph_output", 1.0, ir.DataType.FLOAT16, "output"),
            ("float32_consumer", 1.0, ir.DataType.FLOAT16, "consumer"),
        ]
    )
    def test_skip_rms_unsafe_float32_scale(self, _name, scale, cast_dtype, extra_use):
        scale = None if scale is None else np.full((_D,), scale, dtype=np.float32)
        model, normalized = self._mixed_precision_rms(
            _skip_rms_no_bias, scale, cast_dtype or ir.DataType.FLOAT16
        )
        if cast_dtype is None:
            model.graph.outputs[0] = normalized
            model.graph.remove(next(n for n in model.graph if n.op_type == "Cast"))
        if extra_use == "output":
            model.graph.outputs.append(normalized)
        elif extra_use == "consumer":
            consumer = ir.Node(
                "",
                "Neg",
                [normalized],
                outputs=[ir.Value(type=ir.TensorType(ir.DataType.FLOAT))],
            )
            model.graph.append(consumer)
            model.graph.outputs.append(consumer.outputs[0])
        inputs = {
            k: v.astype(np.float16) for k, v in self._make_inputs("input", "skip").items()
        }
        if scale is None:
            inputs["gamma"] = np.ones((_D,), dtype=np.float32)
        original_output = test_utils.ort_run("Original", model, inputs)
        self.assertEqual(fuse_skip_rms_normalization(model), 0)
        test_utils.assert_allclose(
            test_utils.ort_run("Unfused", model, inputs), original_output
        )

    @parameterized.expand(["input", "skip", "gamma", "bias"])
    def test_skip_rms_mismatched_dtype(self, name):
        model = self._build(
            _skip_rms_post_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D],
            output_types=[self._3D],
        )
        value = next(v for v in model.graph.inputs if v.name == name)
        value.type = ir.TensorType(ir.DataType.FLOAT16)
        self.assertEqual(fuse_skip_rms_normalization(model), 0)

    def test_skip_rms_float16_scale(self):
        model = self._build(
            _skip_rms_no_bias,
            input_types=[FLOAT16[_B, _S, _D], FLOAT16[_B, _S, _D], FLOAT16[_D]],
            output_types=[FLOAT16[_B, _S, _D]],
        )
        inputs = {
            k: v.astype(np.float16)
            for k, v in self._make_inputs("input", "skip", "gamma").items()
        }
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_rms_normalization, expected_count=1
        )

    def test_optimize_for_ort_gemma_skip_rms(self):
        model = self._build(
            _gemma_skip_rms,
            input_types=[FLOAT16[_B, _S, _D], FLOAT16[_B, _S, _D], FLOAT[_D]],
            output_types=[FLOAT16[_B, _S, _D], FLOAT16[_B, _S, _D]],
        )
        gamma = model.graph.inputs[2]
        gamma.const_value = ir.tensor(np.ones((_D,), dtype=np.float32))
        model.graph.inputs.remove(gamma)
        model.graph.register_initializer(gamma)
        inputs = {
            k: v.astype(np.float16) for k, v in self._make_inputs("input", "skip").items()
        }
        original_output = test_utils.ort_run("Original", model, inputs)
        model, counts = optimize_for_ort(model)
        self.assertEqual(counts["skip_rms_normalization"], 1)
        test_utils.assert_allclose(
            test_utils.ort_run("Optimized", model, inputs), original_output
        )
        fused = next(n for n in model.graph if n.op_type == "SkipSimplifiedLayerNormalization")
        self.assertTrue(all(v.dtype == ir.DataType.FLOAT16 for v in fused.inputs))

    # ---- Skip Layer Norm positive tests ----

    def test_skip_ln_no_bias(self):
        """Skip + Input → SkipLayerNormalization."""
        model = self._build(
            _skip_ln_no_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D],
            output_types=[self._3D],
        )
        inputs = self._make_inputs("input", "skip", "gamma", "beta")
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_layer_normalization, expected_count=1
        )
        self.assertEqual(self._count_op(model, "SkipLayerNormalization", "com.microsoft"), 1)
        self.assertEqual(self._count_op(model, "LayerNormalization"), 0)

    def test_skip_ln_post_bias(self):
        """(Input + Skip) + Bias → fuses with bias."""
        model = self._build(
            _skip_ln_post_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D, self._1D],
            output_types=[self._3D],
        )
        inputs = self._make_inputs("input", "skip", "gamma", "beta", "bias")
        self._check_numerical_equivalence(
            model, inputs, fuse_skip_layer_normalization, expected_count=1
        )
        self.assertEqual(self._count_op(model, "SkipLayerNormalization", "com.microsoft"), 1)

    # ---- Negative tests ----

    def test_no_skip_add_no_fusion(self):
        """No Add before norm → rule should not match."""
        model = self._build(
            _skip_rms_no_add,
            input_types=[self._3D, self._1D],
            output_types=[self._3D],
        )
        count = fuse_skip_rms_normalization(model)
        self.assertEqual(count, 0)
        self.assertEqual(
            self._count_op(model, "SkipSimplifiedLayerNormalization", "com.microsoft"), 0
        )

    def test_rank2_input_no_fusion(self):
        """Rank-2 input [S, D] → shape check rejects (expects 3D)."""
        model = self._build(
            _skip_rms_no_bias,
            input_types=[FLOAT["S", _D], FLOAT["S", _D], self._1D],
            output_types=[FLOAT["S", _D]],
        )
        count = fuse_skip_rms_normalization(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
