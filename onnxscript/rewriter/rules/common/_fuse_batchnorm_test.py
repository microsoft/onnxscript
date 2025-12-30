# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.checker
import onnx.parser
import parameterized

from onnxscript import ir
from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _fuse_batchnorm


class FuseBatchnormTest(unittest.TestCase):
    def _create_batchnorm_params(self, size: int):
        return [
            onnx.numpy_helper.from_array(
                np.random.randn(size).astype(np.float32), name="gamma"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(size).astype(np.float32), name="beta"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(size).astype(np.float32), name="input_mean"
            ),
            onnx.numpy_helper.from_array(
                np.abs(np.random.randn(size)).astype(np.float32), name="input_var"
            ),
        ]

    @parameterized.parameterized.expand(
        [
            ("bias_false", False),
            ("bias_true", True),
        ]
    )
    def test_fuse_batchnorm_convtranspose(self, _: str, convtranspose_bias: bool):
        convtranspose_inputs = "X, W"
        parameters = (
            "float[32, 64, 3, 3] W, "
            "float[64] gamma, "
            "float[64] beta, "
            "float[64] input_mean, "
            "float[64] input_var"
        )
        if convtranspose_bias:
            parameters += ", float[64] B"
            convtranspose_inputs += ", B"

        model_proto = onnx.parser.parse_model(f"""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X) => (float [N, ?, ?, ?] Y)
            <{parameters}>
            {{
                X1 = ConvTranspose({convtranspose_inputs})
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }}
        """)
        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(32, 64, 3, 3).astype(np.float32), name="W"
            ),
            *self._create_batchnorm_params(size=64),
        ]
        if convtranspose_bias:
            initializers.append(
                onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), name="B")
            )
        model_proto.graph.initializer.extend(initializers)

        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)

        # Apply rule
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # Check that BatchNorm was fused
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

        # Check inference
        testing.assert_numerically_equal(
            model_proto, model, (np.random.rand(1, 32, 14, 16).astype(np.float32),)
        )

        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    @parameterized.parameterized.expand(
        [
            ("bias_false", False),
            ("bias_true", True),
        ]
    )
    def test_fuse_batchnorm_conv(self, _: str, conv_bias: bool):
        conv_inputs = "X, W"
        parameters = (
            "float[64, 32, 3, 3] W, "
            "float[64] gamma, "
            "float[64] beta, "
            "float[64] input_mean, "
            "float[64] input_var"
        )
        if conv_bias:
            parameters += ", float[64] B"
            conv_inputs += ", B"

        model_proto = onnx.parser.parse_model(f"""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X) => (float [N, ?, ?, ?] Y)
            <{parameters}>
            {{
                X1 = Conv({conv_inputs})
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }}
        """)
        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(64, 32, 3, 3).astype(np.float32), name="W"
            ),
            *self._create_batchnorm_params(size=64),
        ]
        if conv_bias:
            initializers.append(
                onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), name="B")
            )
        model_proto.graph.initializer.extend(initializers)

        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)

        # Apply rule
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # Check that BatchNorm was fused
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

        # Check inference
        testing.assert_numerically_equal(
            model_proto, model, (np.random.rand(1, 32, 14, 16).astype(np.float32),)
        )

        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    @parameterized.parameterized.expand(
        [
            ("bias_false_transB_0", False, 0),
            ("bias_true_transB_0", True, 0),
            ("bias_false_transB_1", False, 1),
            ("bias_true_transB_1", True, 1),
        ]
    )
    def test_fuse_batchnorm_gemm(self, _: str, gemm_bias: bool, transB: int):
        gemm_inputs = "X, W"
        parameters = (
            f"float{'[64, 32]' if transB else '[32, 64]'} W, "
            "float[64] gamma, "
            "float[64] beta, "
            "float[64] input_mean, "
            "float[64] input_var"
        )

        if gemm_bias:
            parameters += ", float[64] B"
            gemm_inputs += ", B"

        model_proto = onnx.parser.parse_model(f"""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32] X) => (float [N, ?] Y)
            <{parameters}>
            {{
                X1 = Gemm<transB={transB}>({gemm_inputs})
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }}
        """)
        weights = np.random.randn(32, 64).astype(np.float32)
        if transB:
            weights = weights.T

        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(weights, name="W"),
            *self._create_batchnorm_params(size=64),
        ]
        if gemm_bias:
            initializers.append(
                onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), name="B")
            )
        model_proto.graph.initializer.extend(initializers)

        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)

        # Apply rule
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # Check that BatchNorm was fused
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 1)

        # Check inference
        testing.assert_numerically_equal(
            model_proto, model, (np.random.rand(1, 32).astype(np.float32),)
        )

        output_model_proto = ir.serde.serialize_model(model)
        onnx.checker.check_model(output_model_proto, True)

    def test_fuse_batchnorm_non_initializers(self):
        model_proto = onnx.parser.parse_model("""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X, float[64, 32, 3, 3] W, float[64] B,
                        float[64] gamma, float[64] beta, float[64] input_var,
                        float[64] input_mean) => (float [N, ?, ?, ?] Y)
            {
                X1 = Conv(X, W, B)
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }
        """)
        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # No changes were applied
        self.assertEqual(count, 0)

    def test_fuse_batchnorm_graph_inputs(self):
        model_proto = onnx.parser.parse_model("""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X, float[64, 32, 3, 3] W) => (float [N, ?, ?, ?] Y)
            {
                X1 = Conv(X, W)
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }
        """)
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(64, 32, 3, 3).astype(np.float32), name="W"
            ),
            *self._create_batchnorm_params(size=64),
        ]
        model_proto.graph.initializer.extend(initializers)
        onnx.checker.check_model(model_proto, True)

        model = ir.serde.deserialize_model(model_proto)
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # No changes were applied as W is a graph input
        self.assertEqual(count, 0)

    def test_fuse_batchnorm_does_not_collide_names_with_same_parent_node(self):
        model_proto = onnx.parser.parse_model("""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X) => (float [N, ?, ?, ?] Y1, float [N, ?, ?, ?] Y2)
            {
                X1 = MaxPool<kernel_shape=[3,3]>(X)
                X2 = Conv(X1, W1)
                Y1 = BatchNormalization(X2, gamma_64, beta_64, input_mean_64, input_var_64)
                X3 = Conv(X1, W2)
                Y2 = BatchNormalization(X3, gamma_256, beta_256, input_mean_256, input_var_256)
            }
        """)
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(64, 32, 3, 3).astype(np.float32), name="W1"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(64).astype(np.float32), name="gamma_64"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(64).astype(np.float32), name="beta_64"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(64).astype(np.float32), name="input_mean_64"
            ),
            onnx.numpy_helper.from_array(
                np.abs(np.random.randn(64)).astype(np.float32), name="input_var_64"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(256, 32, 3, 3).astype(np.float32), name="W2"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(256).astype(np.float32), name="gamma_256"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(256).astype(np.float32), name="beta_256"
            ),
            onnx.numpy_helper.from_array(
                np.random.randn(256).astype(np.float32), name="input_mean_256"
            ),
            onnx.numpy_helper.from_array(
                np.abs(np.random.randn(256)).astype(np.float32), name="input_var_256"
            ),
        ]
        model_proto.graph.initializer.extend(initializers)
        onnx.checker.check_model(model_proto, True)
        model = ir.serde.deserialize_model(model_proto)
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # Applied twice, once for each BatchNorm
        self.assertEqual(count, 2)
        # it should have different bias names for the two fused Conv nodes
        conv_nodes = [node for node in model.graph if node.op_type == "Conv"]
        self.assertEqual(len(conv_nodes), 2)
        bias_names_1 = conv_nodes[0].inputs[2].name
        bias_names_2 = conv_nodes[1].inputs[2].name
        self.assertNotEqual(bias_names_1, bias_names_2)


if __name__ == "__main__":
    unittest.main()
