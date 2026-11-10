# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import onnx.parser
import onnx_ir as ir
import parameterized

from onnxscript.rewriter.ort_fusions import instance_to_group_normalization, optimize_for_ort
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run


class OptimizeGroupNormalizationTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (opset, groups, channels, dtype, affine_channels)
            for opset in (17, 20, 21, 23)
            for groups, channels in ((32, 320), (32, 32), (8, 40))
            for dtype in ("float", "float16")
            for affine_channels in (channels, 1)
        ]
    )
    def test_optimize_for_ort_uses_supported_group_normalization(
        self, opset, groups, channels, dtype, affine_channels
    ):
        model_proto = onnx.parser.parse_model(
            f"""
                <ir_version: 9, opset_import: [ "" : {opset}]>
                agraph ({dtype}[2, {channels}, 4, 4] image)
                    => ({dtype}[2, {channels}, 4, 4] output)
                {{
                    adjusted_input_shape = Constant<value: tensor = int64[3] {{0, {groups}, -1}}>()
                    image_reshape = Reshape (image, adjusted_input_shape)
                    instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                    original_input_shape = Constant<value: tensor = int64[4] {{2, {channels}, 4, 4}}>()
                    instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                    mul_output = Mul (instance_norm_reshape, weight_full)
                    output = Add (mul_output, bias_full)
                }}
                """
        )
        np_dtype = np.float32 if dtype == "float" else np.float16
        rng = np.random.default_rng(0)
        initializers = {
            "weight_for_norm": np.ones(groups, dtype=np_dtype),
            "bias_for_norm": np.zeros(groups, dtype=np_dtype),
            "weight_full": rng.random((affine_channels, 1, 1)).astype(np_dtype),
            "bias_full": rng.random((affine_channels, 1, 1)).astype(np_dtype),
        }
        for name, value in initializers.items():
            model_proto.graph.initializer.append(
                ir.serde.serialize_tensor(ir.tensor(value, name=name))
            )
        model = ir.from_proto(model_proto)
        inputs = {"image": rng.standard_normal((2, channels, 4, 4)).astype(np_dtype)}
        expected = ort_run("original_group_norm", model, inputs)

        optimize_for_ort(model)

        self.assertEqual(model.opset_imports[""], opset)
        self.assertNotIn(
            ("com.microsoft", "GroupNorm"),
            {(node.domain, node.op_type) for node in model.graph},
        )
        group_norms = [node for node in model.graph if node.op_type == "GroupNormalization"]
        if opset >= 21 and affine_channels == channels:
            self.assertEqual(len(group_norms), 1)
            group_norm = group_norms[0]
            self.assertEqual(group_norm.domain, "")
            self.assertEqual(group_norm.attributes["num_groups"].as_int(), groups)
            self.assertAlmostEqual(group_norm.attributes["epsilon"].as_float(), 1e-6)
            self.assertEqual(group_norm.inputs[1].shape, ir.Shape([channels]))
            self.assertEqual(group_norm.inputs[2].shape, ir.Shape([channels]))
        else:
            self.assertEqual(len(group_norms), 0)
            self.assertIn("InstanceNormalization", {node.op_type for node in model.graph})
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        assert_allclose(ort_run("optimized_group_norm", model, inputs), expected)


class ReplaceInstanceNormWithGroupNormTest(unittest.TestCase):
    def _set_up_model_initializers(
        self,
        model,
        weight_for_norm_value,
        weight_for_norm_shape,
        bias_for_norm_value,
        bias_for_norm_shape,
        weight_full_value,
        weight_full_shape,
        bias_full_value,
        bias_full_shape,
    ):
        """Set up the model initializers for the test."""
        model.graph.initializer.extend(
            [
                onnx.helper.make_tensor(
                    "weight_for_norm",
                    onnx.TensorProto.FLOAT16,
                    weight_for_norm_shape,
                    weight_for_norm_value,
                ),
                onnx.helper.make_tensor(
                    "bias_for_norm",
                    onnx.TensorProto.FLOAT16,
                    bias_for_norm_shape,
                    bias_for_norm_value,
                ),
                onnx.helper.make_tensor(
                    "weight_full",
                    onnx.TensorProto.FLOAT16,
                    weight_full_shape,
                    weight_full_value,
                ),
                onnx.helper.make_tensor(
                    "bias_full",
                    onnx.TensorProto.FLOAT16,
                    bias_full_shape,
                    bias_full_value,
                ),
            ]
        )

    def test_simulated_instance_norm_is_replaced_by_group_norm(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        # plus 2 in model constants
        self.assertEqual(len(model.graph), 10)

    def test_instance_norm_with_non_one_weight_for_norm_should_remain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.random.rand(32).astype(np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_zero_b_should_remain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.random.rand(32).astype(np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_broadcasted_weight_full_should_remain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_broadcasted_bias_full_should_remain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_rank_not_4_should_remain(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_weight_full_having_multiple_not_one_dim_should_remain(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 2, 3).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 2, 3],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_bias_full_having_multiple_not_one_dim_should_remain(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 2, 3).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 2, 3],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_not_0_g_negative_1_shape_of_adjusted_input_shape_should_remain(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 16, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_equal_of_image_shape_and_original_input_shape_should_remain(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {2, 320, 64, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                output = Add (mul_output, bias_full)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)
        self._set_up_model_initializers(
            model_proto,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
