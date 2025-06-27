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
from onnxscript.rewriter.ort_fusions.gelu import fuse_gelu


class GeluFusionTest(unittest.TestCase):
    def test_gelu_fusion(self):
        _sqrt_two_over_pi = math.sqrt(2.0 / math.pi)

        @script()
        def gelu_model(x):
            # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
            t1 = op.Pow(x, 3)
            t2 = op.Mul(0.044715, t1)
            t3 = op.Add(x, t2)

            t4 = op.Mul(_sqrt_two_over_pi, t3)
            t5 = op.Tanh(t4)
            t6 = op.Add(t5, 1)
            t7 = op.Mul(0.5, t6)
            result = op.Mul(x, t7)
            return result

        model_proto = gelu_model.to_model_proto(
            input_types=[FLOAT[10]], output_types=[FLOAT[10]]
        )
        model = ir.serde.deserialize_model(model_proto)

        # Eliminate redundant CastLike ops:
        optimize(model)

        input = {"x": np.random.randn(10).astype(np.float32)}
        original_output = test_utils.ort_run("Original", model, input)

        fuse_gelu(model)
        remove_unused_nodes(model)

        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph.node(0).op_type, "FastGelu")

        optimized_output = test_utils.ort_run("Optimized", model, input)
        test_utils.assert_allclose(original_output, optimized_output)


if __name__ == "__main__":
    unittest.main()
