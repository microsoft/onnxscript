# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx
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
            ("bias_false_group1", False, 1),
            ("bias_true_group1", True, 1),
            ("bias_false_group4", False, 4),
            ("bias_true_group4", True, 4),
        ]
    )
    def test_fuse_batchnorm_convtranspose(self, _: str, convtranspose_bias: bool, group: int):
        # ConvTranspose weight: [in_channels, out_channels/group, kH, kW]
        out_channels = 64 * group
        convtranspose_inputs = "X, W"
        parameters = (
            f"float[32, 64, 3, 3] W, "
            f"float[{out_channels}] gamma, "
            f"float[{out_channels}] beta, "
            f"float[{out_channels}] input_mean, "
            f"float[{out_channels}] input_var"
        )
        if convtranspose_bias:
            parameters += f", float[{out_channels}] B"
            convtranspose_inputs += ", B"

        model_proto = onnx.parser.parse_model(f"""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 16] X) => (float [N, ?, ?, ?] Y)
            <{parameters}>
            {{
                X1 = ConvTranspose<group={group}>({convtranspose_inputs})
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }}
        """)
        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(32, 64, 3, 3).astype(np.float32), name="W"
            ),
            *self._create_batchnorm_params(size=out_channels),
        ]
        if convtranspose_bias:
            initializers.append(
                onnx.numpy_helper.from_array(
                    np.random.randn(out_channels).astype(np.float32), name="B"
                )
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
            ("bias_false_group1", False, 1),
            ("bias_true_group1", True, 1),
            ("bias_false_group2", False, 2),
            ("bias_true_group2", True, 2),
        ]
    )
    def test_fuse_batchnorm_conv(self, _: str, conv_bias: bool, group: int):
        # Conv weight: [out_channels, in_channels/group, kH, kW]
        in_channels_per_group = 32 // group
        conv_inputs = "X, W"
        parameters = (
            f"float[64, {in_channels_per_group}, 3, 3] W, "
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
                X1 = Conv<group={group}>({conv_inputs})
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }}
        """)
        # Add initializers
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(64, in_channels_per_group, 3, 3).astype(np.float32), name="W"
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

    def test_fuse_batchnorm_convtranspose_grouped_invalid_skipped(self):
        """Fusion is skipped when in_channels is not divisible by group (semantically invalid model)."""
        # in_channels=32 is not divisible by group=3, the ONNX checker won't catch this.
        model_proto = onnx.parser.parse_model("""
            < ir_version: 7, opset_import: ["" : 17] >
            test_model (float[N, 32, 14, 14] X) => (float[N, ?, ?, ?] Y)
            <float[32, 64, 3, 3] W,
             float[192] gamma, float[192] beta, float[192] input_mean, float[192] input_var>
            {
                X1 = ConvTranspose<group=3>(X, W)
                Y = BatchNormalization(X1, gamma, beta, input_mean, input_var)
            }
        """)
        initializers = [
            onnx.numpy_helper.from_array(
                np.random.randn(32, 64, 3, 3).astype(np.float32), name="W"
            ),
            *self._create_batchnorm_params(size=192),
        ]
        model_proto.graph.initializer.extend(initializers)
        model = ir.serde.deserialize_model(model_proto)
        count = _fuse_batchnorm.rules.apply_to_model(model)

        # Fusion must be skipped, applying it would crash on the invalid dimensions.
        self.assertEqual(count, 0)

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
