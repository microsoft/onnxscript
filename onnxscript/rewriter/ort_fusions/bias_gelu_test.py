# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np

import onnxscript
import onnxscript.ir as ir
import onnxscript.rewriter.ort_fusions._test_utils as test_utils
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter.ort_fusions.bias_gelu import fuse_bias_gelu

msft_op = onnxscript.values.Opset("com.microsoft", 1)


class BiasGeluFusionTest(unittest.TestCase):
    def test_bias_gelu_fusion(self):
        @script()
        def bias_gelu_model(x, y):
            gelu_add = op.Add(x, y)
            gelu = msft_op.Gelu(gelu_add)
            return gelu

        model_proto = bias_gelu_model.to_model_proto(
            input_types=[FLOAT[10], FLOAT[10]],
            output_types=[FLOAT[10]],
            ir_version=10,
        )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)

        input = {
            "x": np.random.randn(10).astype(np.float32),
            "y": np.random.randn(10).astype(np.float32),
        }
        original_output = test_utils.ort_run("Original", model, input)

        fuse_bias_gelu(model)
        remove_unused_nodes(model)

        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph.node(0).op_type, "BiasGelu")

        optimized_output = test_utils.ort_run("Optimized", model, input)
        test_utils.assert_allclose(original_output, optimized_output)


if __name__ == "__main__":
    unittest.main()
