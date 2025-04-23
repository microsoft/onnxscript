# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np

import onnx
import onnxscript.ir as ir
import onnxscript.rewriter.ort_fusions._test_utils as test_utils

from onnxscript import opset18 as op
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter.ort_fusions.bias_gelu import fuse_bias_gelu

FLOAT = onnx.TensorProto.FLOAT


class BiasGeluFusionTest(unittest.TestCase):
    def test_gelu_fusion(self):

        model_proto = onnx.helper.make_model(
                onnx.helper.make_graph(
                    [
                        onnx.helper.make_node("Add", ["X", "Y"], ["gelu_add"]),
                        onnx.helper.make_node(
                            "Gelu",
                            ["gelu_add"],
                            ["Z"],
                            domain="com.microsoft",
                        ),
                    ],
                    "name",
                    [
                        onnx.helper.make_tensor_value_info("X", FLOAT, [10]),
                        onnx.helper.make_tensor_value_info("Y", FLOAT, [10]),
                    ],
                    [onnx.helper.make_tensor_value_info("Z", FLOAT, [10])],
                ),
                opset_imports=[
                    onnx.helper.make_opsetid("", 18),
                    onnx.helper.make_opsetid("com.microsoft", 1),
                ],
            )
        model = ir.serde.deserialize_model(model_proto)
        optimize(model)

        input = {"X": np.random.randn(10).astype(np.float32), "Y": np.random.randn(10).astype(np.float32)}
        original_output = test_utils.ort_run("Original", model, input)

        fuse_bias_gelu(model)
        remove_unused_nodes(model)

        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph.node(0).op_type, "BiasGelu")

        optimized_output = test_utils.ort_run("Optimized", model, input)
        test_utils.assert_allclose(original_output, optimized_output)


if __name__ == "__main__":
    unittest.main()