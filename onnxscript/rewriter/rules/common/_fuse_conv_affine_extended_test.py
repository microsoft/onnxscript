# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for ConvAffineFusion and AffineConvFusion rules.

Adds coverage for: non-constant weight (negative), non-scalar scale (negative),
padded pre-conv affine (negative), non-constant bias (negative), positive with
numerical validation.
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter import rewrite, testing
from onnxscript.rewriter.rules.common import (
    affine_conv_fusion_rule,
    conv_affine_fusion_rule,
)

# Constants used in @script() models
_W_ONES = np.ones((3, 3, 3, 3), dtype=np.float32)
_B_ONES = np.ones((3,), dtype=np.float32)


class FuseConvAffineExtendedTest(unittest.TestCase):
    """Extended tests for ConvAffineFusion and AffineConvFusion."""

    def _clone(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    # --- Negative: non-constant weight ---

    def test_conv_affine_non_constant_weight_no_fusion(self):
        """Non-constant weight (graph input) -> check rejects (w must be constant)."""

        @script()
        def model_fn(
            x: FLOAT[1, 3, 32, 32],
            w: FLOAT[3, 3, 3, 3],
        ) -> FLOAT[1, 3, 32, 32]:
            b = op.Constant(value=_B_ONES)
            scale = op.Constant(value_float=2.0)
            offset = op.Constant(value_float=3.0)
            conv_out = op.Conv(x, w, b, pads=[1, 1, 1, 1])
            return (conv_out * scale) + offset

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: non-scalar scale ---

    def test_conv_affine_non_scalar_scale_no_fusion(self):
        """Vector scale -> check rejects (scale must be scalar)."""

        @script()
        def model_fn(x: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
            w = op.Constant(value=_W_ONES)
            b = op.Constant(value=_B_ONES)
            scale = op.Constant(value_floats=[2.0, 3.0, 4.0])  # vector, not scalar
            offset = op.Constant(value_float=3.0)
            conv_out = op.Conv(x, w, b, pads=[1, 1, 1, 1])
            return (conv_out * scale) + offset

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: non-constant bias ---

    def test_conv_affine_non_constant_bias_no_fusion(self):
        """Non-constant bias (graph input) -> check rejects (b must be constant)."""

        @script()
        def model_fn(
            x: FLOAT[1, 3, 32, 32],
            b: FLOAT[3],
        ) -> FLOAT[1, 3, 32, 32]:
            w = op.Constant(value=_W_ONES)
            scale = op.Constant(value_float=2.0)
            offset = op.Constant(value_float=3.0)
            conv_out = op.Conv(x, w, b, pads=[1, 1, 1, 1])
            return (conv_out * scale) + offset

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: pre-conv affine with padding ---

    def test_affine_conv_with_padding_no_fusion(self):
        """Pre-conv affine + padded Conv -> AffineConvFusion must NOT match.

        AffineConvFusion pattern requires pads=[0,0,0,0].
        """

        @script()
        def model_fn(x: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
            w = op.Constant(value=_W_ONES)
            b = op.Constant(value=_B_ONES)
            scale = op.Constant(value_float=2.0)
            offset = op.Constant(value_float=3.0)
            affine = (x * scale) + offset
            return op.Conv(affine, w, b, pads=[1, 1, 1, 1])  # non-zero pads

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[affine_conv_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Positive: conv-affine fusion with all-constant operands ---

    def test_conv_affine_positive_fuses(self):
        """Standard Conv -> Mul(scalar) -> Add(scalar) with constant w, b fuses correctly."""

        @script()
        def model_fn(x: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
            w = op.Constant(value=_W_ONES)
            b = op.Constant(value=_B_ONES)
            scale = op.Constant(value_float=2.0)
            offset = op.Constant(value_float=3.0)
            conv_out = op.Conv(x, w, b, pads=[1, 1, 1, 1])
            return (conv_out * scale) + offset

        model_proto = model_fn.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        # Mul and Add should be fused — neither should remain in the graph
        rewritten_ops = [n.op_type for n in rewritten.graph]
        self.assertNotIn("Mul", rewritten_ops)
        self.assertNotIn("Add", rewritten_ops)
        self.assertIn("Conv", rewritten_ops)

        # Numerical validation
        rng = np.random.default_rng(42)
        inputs = [rng.random((1, 3, 32, 32), dtype=np.float32)]
        testing.assert_numerically_equal(model, rewritten, inputs)


if __name__ == "__main__":
    unittest.main()
