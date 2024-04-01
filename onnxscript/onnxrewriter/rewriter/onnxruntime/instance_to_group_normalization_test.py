import unittest

import numpy as np
import onnx.parser

from onnxrewriter.ir import irbuilder
from onnxrewriter.rewriter.onnxruntime import instance_to_group_normalization


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
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 1)
        # plus 2 in model constants
        self.assertEqual(len(ir.graph.nodes), 10)

    def test_instance_norm_with_non_one_weight_for_norm_should_remain(self):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_zero_b_should_remain(self):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_broadcasted_weight_full_should_remain(self):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_broadcasted_bias_full_should_remain(self):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_rank_not_4_should_remain(self):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_weight_full_having_multiple_not_one_dim_should_remain(
        self,
    ):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 2, 3],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_bias_full_having_multiple_not_one_dim_should_remain(
        self,
    ):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 2, 3],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_not_0_g_negative_1_shape_of_adjusted_input_shape_should_remain(
        self,
    ):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)

    def test_instance_norm_with_non_equal_of_image_shape_and_original_input_shape_should_remain(
        self,
    ):
        model = onnx.parser.parse_model(
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
            model,
            weight_for_norm_value,
            [32],
            bias_for_norm_value,
            [32],
            weight_full_value,
            [320, 1, 1],
            bias_full_value,
            [320, 1, 1],
        )

        ir = irbuilder.build_ir(model)
        count = instance_to_group_normalization.rules.apply_to_model(ir)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
