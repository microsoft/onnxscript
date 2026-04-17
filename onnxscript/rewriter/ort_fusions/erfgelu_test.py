# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import unittest

import numpy as np
import onnx_ir as ir

import onnxscript.rewriter.ort_fusions._test_utils as test_utils
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter.ort_fusions.erfgelu import fuse_erfgelu

_SQRT_TWO = math.sqrt(2.0)


class ErfGeluFusionTest(unittest.TestCase):
    """Tests for erf-based GELU fusion patterns in erfgelu.py.

    Pattern 1: 0.5 * (x * (erf(x / sqrt(2)) + 1))
    Pattern 2: x * (0.5 * (erf(x / sqrt(2)) + 1))
    """

    def _check_fusion(self, model, input):
        original_output = test_utils.ort_run("Original", model, input)
        fuse_erfgelu(model)
        remove_unused_nodes(model)
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph.node(0).op_type, "Gelu")
        self.assertEqual(model.graph.node(0).domain, "com.microsoft")
        optimized_output = test_utils.ort_run("Optimized", model, input)
        test_utils.assert_allclose(original_output, optimized_output)

    def _check_no_fusion(self, model):
        node_count_before = len(model.graph)
        fuse_erfgelu(model)
        remove_unused_nodes(model)
        self.assertEqual(len(model.graph), node_count_before)
        self.assertTrue(
            all(node.op_type != "Gelu" for node in model.graph),
            "Gelu node should not be present after failed fusion",
        )

    def _build_model(self, script_fn, shape):
        model_proto = script_fn.to_model_proto(
            input_types=[FLOAT[shape]], output_types=[FLOAT[shape]]
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)
        return model

    def test_pattern1_half_times_x_times_erf_plus_one(self):
        """Pattern 1: 0.5 * (x * (erf(x / sqrt(2)) + 1))"""

        @script()
        def erf_gelu_p1(x):
            t1 = op.Div(x, _SQRT_TWO)
            t2 = op.Erf(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(x, t3)
            return op.Mul(0.5, t4)

        model = self._build_model(erf_gelu_p1, 10)
        input = {"x": np.random.randn(10).astype(np.float32)}
        self._check_fusion(model, input)

    def test_pattern2_x_times_half_times_erf_plus_one(self):
        """Pattern 2: x * (0.5 * (erf(x / sqrt(2)) + 1))"""

        @script()
        def erf_gelu_p2(x):
            t1 = op.Div(x, _SQRT_TWO)
            t2 = op.Erf(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(0.5, t3)
            return op.Mul(x, t4)

        model = self._build_model(erf_gelu_p2, 10)
        input = {"x": np.random.randn(10).astype(np.float32)}
        self._check_fusion(model, input)

    def test_multidimensional_input(self):
        """Verify fusion works with 3D inputs (batch, seq, hidden)."""

        @script()
        def erf_gelu_3d(x):
            t1 = op.Div(x, _SQRT_TWO)
            t2 = op.Erf(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(x, t3)
            return op.Mul(0.5, t4)

        model = self._build_model(erf_gelu_3d, (2, 4, 8))
        input = {"x": np.random.randn(2, 4, 8).astype(np.float32)}
        self._check_fusion(model, input)

    def test_no_fusion_without_erf(self):
        """Replacing Erf with Tanh should not match the erf-gelu pattern."""

        @script()
        def not_erf_gelu(x):
            t1 = op.Div(x, _SQRT_TWO)
            t2 = op.Tanh(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(x, t3)
            return op.Mul(0.5, t4)

        model = self._build_model(not_erf_gelu, 10)
        self._check_no_fusion(model)

    def test_no_fusion_wrong_divisor(self):
        """Using a divisor other than sqrt(2) should not match."""

        @script()
        def wrong_divisor(x):
            t1 = op.Div(x, 2.0)
            t2 = op.Erf(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(x, t3)
            return op.Mul(0.5, t4)

        model = self._build_model(wrong_divisor, 10)
        self._check_no_fusion(model)

    def test_no_fusion_wrong_scale(self):
        """Using 0.3 instead of 0.5 should not match."""

        @script()
        def wrong_scale(x):
            t1 = op.Div(x, _SQRT_TWO)
            t2 = op.Erf(t1)
            t3 = op.Add(t2, 1.0)
            t4 = op.Mul(x, t3)
            return op.Mul(0.3, t4)

        model = self._build_model(wrong_scale, 10)
        self._check_no_fusion(model)


if __name__ == "__main__":
    unittest.main()
