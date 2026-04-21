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

from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize
from onnxscript.rewriter.ort_fusions.skip_normalization import (
    fuse_skip_layer_normalization,
    fuse_skip_rms_normalization,
)

_B, _S, _D = 2, 8, 16
_EPS_F = ir.tensor(np.array([1e-6], dtype=np.float32))


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

    _3D = FLOAT["B", "S", _D]
    _1D = FLOAT[_D]

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
        count = fuse_skip_rms_normalization(model)
        self.assertGreater(count, 0)
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
        count = fuse_skip_rms_normalization(model)
        self.assertGreater(count, 0)
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
        count = fuse_skip_rms_normalization(model)
        self.assertGreater(count, 0)
        self.assertEqual(
            self._count_op(model, "SkipSimplifiedLayerNormalization", "com.microsoft"), 1
        )

    # ---- Skip Layer Norm positive tests ----

    def test_skip_ln_no_bias(self):
        """Skip + Input → SkipLayerNormalization."""
        model = self._build(
            _skip_ln_no_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D],
            output_types=[self._3D],
        )
        count = fuse_skip_layer_normalization(model)
        self.assertGreater(count, 0)
        self.assertEqual(self._count_op(model, "SkipLayerNormalization", "com.microsoft"), 1)
        self.assertEqual(self._count_op(model, "LayerNormalization"), 0)

    def test_skip_ln_post_bias(self):
        """(Input + Skip) + Bias → fuses with bias."""
        model = self._build(
            _skip_ln_post_bias,
            input_types=[self._3D, self._3D, self._1D, self._1D, self._1D],
            output_types=[self._3D],
        )
        count = fuse_skip_layer_normalization(model)
        self.assertGreater(count, 0)
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
