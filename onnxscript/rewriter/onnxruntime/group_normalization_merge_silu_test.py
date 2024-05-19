import unittest

import numpy as np
import onnx.parser

from onnxscript import ir
from onnxscript.rewriter.onnxruntime import (
    group_normalization_merge_silu,
    instance_to_group_normalization,
)


class ReplaceInstanceNormWithGroupNormTest(unittest.TestCase):
    def test_group_norm_with_silu_submodule_is_replaced_by_group_norm(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: ["" : 17, "pkg.torch230a0git77ef9d4" : 1, "com.microsoft" : 1]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                group_norm = com.microsoft.GroupNorm <activation=0, channels_last=1, epsilon=0.000001, groups=32>(image, weight, bias)
                transposed = Transpose <perm=[0, 3, 1, 2]>(group_norm)
                output = pkg.torch230a0git77ef9d4.torch_nn_modules_activation_SiLU_time_embedding_act_19 (transposed)
            }
            <domain: "pkg.torch230a0git77ef9d4", opset_import: ["" : 17]>
            torch_nn_modules_activation_SiLU_time_embedding_act_19 (transposed) => (output)
            {
                _to_copy_38 = Cast <to: int = 1> (transposed)
                sigmoid_18 = Sigmoid (_to_copy_38)
                mul_26 = Mul (_to_copy_38, sigmoid_18)
                output = Cast <to: int = 10> (mul_26)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_value = np.random.rand(320, 1, 1).astype(np.float16)
        model_proto.graph.initializer.extend(
            [
                onnx.helper.make_tensor(
                    "weight",
                    onnx.TensorProto.FLOAT16,
                    weight_value.shape,
                    weight_value,
                ),
                onnx.helper.make_tensor(
                    "bias",
                    onnx.TensorProto.FLOAT16,
                    bias_value.shape,
                    bias_value,
                ),
            ]
        )

        model = ir.serde.deserialize_model(model_proto)
        count = group_normalization_merge_silu.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        # plus 2 in model constants
        self.assertEqual(len(model.graph), 2)

    def test_simulated_instance_norm_is_replaced_by_group_norm_silu(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "pkg.torch230a0git77ef9d4" : 1]>
            agraph (float[1, 320, 128, 128] image) => (float[1, 4, 512, 64] output)
            {
                adjusted_input_shape = Constant<value: tensor = int64[3] {0, 32, -1}>()
                image_reshape = Reshape (image, adjusted_input_shape)
                instance_norm = InstanceNormalization <epsilon=0.000001>(image_reshape, weight_for_norm, bias_for_norm)
                original_input_shape = Constant<value: tensor = int64[4] {1, 320, 128, 128}>()
                instance_norm_reshape = Reshape (instance_norm, original_input_shape)
                mul_output = Mul (instance_norm_reshape, weight_full)
                add_output = Add (mul_output, bias_full)
                output = pkg.torch230a0git77ef9d4.torch_nn_modules_activation_SiLU_time_embedding_act_19 (add_output)
            }
            <domain: "pkg.torch230a0git77ef9d4", opset_import: ["" : 17]>
            torch_nn_modules_activation_SiLU_time_embedding_act_19 (add_output) => (output)
            {
                _to_copy_38 = Cast <to: int = 1> (add_output)
                sigmoid_18 = Sigmoid (_to_copy_38)
                mul_26 = Mul (_to_copy_38, sigmoid_18)
                output = Cast <to: int = 10> (mul_26)
            }
        """
        )
        # Use inserted initializers to avoid manually coding the large constants
        weight_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        bias_full_value = np.random.rand(320, 1, 1).astype(np.float16)
        weight_for_norm_value = np.ones(32, dtype=np.float16)
        bias_for_norm_value = np.zeros(32, dtype=np.float16)

        model_proto.graph.initializer.extend(
            [
                onnx.helper.make_tensor(
                    "weight_for_norm",
                    onnx.TensorProto.FLOAT16,
                    weight_for_norm_value.shape,
                    weight_for_norm_value,
                ),
                onnx.helper.make_tensor(
                    "bias_for_norm",
                    onnx.TensorProto.FLOAT16,
                    bias_for_norm_value.shape,
                    bias_for_norm_value,
                ),
                onnx.helper.make_tensor(
                    "weight_full",
                    onnx.TensorProto.FLOAT16,
                    weight_full_value.shape,
                    weight_full_value,
                ),
                onnx.helper.make_tensor(
                    "bias_full",
                    onnx.TensorProto.FLOAT16,
                    bias_full_value.shape,
                    bias_full_value,
                ),
            ]
        )

        model = ir.serde.deserialize_model(model_proto)
        count = instance_to_group_normalization.rules.apply_to_model(model)
        count += group_normalization_merge_silu.rules.apply_to_model(model)
        self.assertEqual(count, 2)
        # plus 2 in model constants
        self.assertEqual(len(model.graph), 10)


if __name__ == "__main__":
    unittest.main()
