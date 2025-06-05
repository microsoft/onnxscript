# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import parameterized

import onnxscript
import onnxscript.ir as ir
import onnxscript.rewriter.ort_fusions._test_utils as test_utils
from onnxscript import FLOAT, OnnxFunction, script
from onnxscript import opset20 as op
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter.ort_fusions.bias_gelu import fuse_bias_gelu

msft_op = onnxscript.values.Opset("com.microsoft", 1)


@script()
def _test_script_onnx_default(x: FLOAT[10], y: FLOAT[10]) -> FLOAT[10]:
    gelu_add = op.Add(x, y)
    return op.Gelu(gelu_add)


@script()
def _test_script_onnx_none(x: FLOAT[10], y: FLOAT[10]) -> FLOAT[10]:
    gelu_add = op.Add(x, y)
    return op.Gelu(gelu_add, approximate="none")


@script()
def _test_script_msft_op(x: FLOAT[10], y: FLOAT[10]) -> FLOAT[10]:
    gelu_add = op.Add(x, y)
    return msft_op.Gelu(gelu_add)


class BiasGeluFusionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("with_onnx_op_default", _test_script_onnx_default),
            ("with_onnx_op_none", _test_script_onnx_none),
            ("with_contrib_op", _test_script_msft_op),
        ]
    )
    def test_bias_gelu_fusion(self, _: str, test_data_constructor: OnnxFunction):
        model_proto = test_data_constructor.to_model_proto()
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
