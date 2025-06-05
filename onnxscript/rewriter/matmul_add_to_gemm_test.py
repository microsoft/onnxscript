# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import matmul_add_to_gemm, testing


class MatMulAddToMatMulTest(unittest.TestCase):
    def _get_test_model(self, x_shape, w_shape, b_shape):
        output_shape = "[" + ", ".join("?" for _ in x_shape) + "]"
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float{x_shape!s} X, float{w_shape!s} W, float{b_shape!s} B) => (float{output_shape} Y)
            {{
                x_matmul = MatMul(X, W)
                Y = Add(x_matmul, B)
            }}
        """
        )
        return model_proto

    def test_matmul_add_to_gemm_inputs(self):
        model_proto = self._get_test_model(x_shape=[256, 512], w_shape=[512, 64], b_shape=[64])
        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)
        count = matmul_add_to_gemm.rule.apply_to_model(model)

        # Check MatMul + Add are fused
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

        # Check inference
        testing.assert_numerically_equal(
            model_proto,
            model,
            (
                np.random.rand(256, 512).astype(np.float32),
                np.random.rand(512, 64).astype(np.float32),
                np.random.rand(
                    64,
                ).astype(np.float32),
            ),
        )

        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    def test_matmul_add_to_gemm_initializers(self):
        model_proto = self._get_test_model(x_shape=[256, 512], w_shape=[512, 64], b_shape=[64])

        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(np.random.rand(512, 64).astype(np.float32), name="W"),
            onnx.numpy_helper.from_array(
                np.random.rand(
                    64,
                ).astype(np.float32),
                name="B",
            ),
        ]
        model_proto.graph.initializer.extend(initializers)
        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)
        count = matmul_add_to_gemm.rule.apply_to_model(model)

        # Check MatMul + Add are fused
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

        # Check inference
        testing.assert_numerically_equal(
            model_proto, model, (np.random.rand(256, 512).astype(np.float32),)
        )

        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    def test_matmul_add_to_gemm_incompatible_shapes(self):
        model_proto = self._get_test_model(
            x_shape=[1, 256, 512], w_shape=[1, 512, 64], b_shape=[64]
        )

        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)

        # Check that the model is unchanged
        count = matmul_add_to_gemm.rule.apply_to_model(model)
        self.assertEqual(count, 0)
        self.assertEqual(len(model.graph), 2)


if __name__ == "__main__":
    unittest.main()
