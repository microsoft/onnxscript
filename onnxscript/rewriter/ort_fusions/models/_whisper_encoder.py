# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A one-layer Whisper encoder model test case, with inputs: audio_features.
This is an onnxscript version of the model.
"""

import numpy as np

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT


def make_model(
    encoder_encoder_embed_positions_weight,
    encoder_encoder_conv1_weight,
    encoder_encoder_conv1_bias,
    encoder_encoder_conv2_weight,
    encoder_encoder_conv2_bias,
    encoder_encoder_layers_0_self_attn_layer_norm_weight,
    encoder_encoder_layers_0_self_attn_layer_norm_bias,
    encoder_encoder_layers_0_self_attn_q_proj_weight,
    encoder_encoder_layers_0_self_attn_q_proj_bias,
    encoder_encoder_layers_0_self_attn_k_proj_weight,
    encoder_encoder_layers_0_self_attn_v_proj_weight,
    encoder_encoder_layers_0_self_attn_v_proj_bias,
    encoder_encoder_layers_0_self_attn_out_proj_weight,
    encoder_encoder_layers_0_self_attn_out_proj_bias,
    encoder_encoder_layers_0_final_layer_norm_weight,
    encoder_encoder_layers_0_final_layer_norm_bias,
    encoder_encoder_layers_0_fc1_weight,
    encoder_encoder_layers_0_fc1_bias,
    encoder_encoder_layers_0_fc2_weight,
    encoder_encoder_layers_0_fc2_bias,
    encoder_encoder_layer_norm_weight,
    encoder_encoder_layer_norm_bias,
):
    @script()
    def main_graph(
        audio_features: FLOAT[1, 80, 3000],
    ) -> FLOAT[1, 1500, 384]:
        val_0 = opset18.Shape(audio_features, end=1, start=0)
        conv1d = opset18.Conv(
            audio_features,
            encoder_encoder_conv1_weight,
            encoder_encoder_conv1_bias,
            group=1,
            pads=[1, 1],
            auto_pad="NOTSET",
            strides=[1],
            dilations=[1],
        )
        val_2 = opset18.Div(conv1d, 1.4142135)
        val_3 = opset18.Erf(val_2)
        val_5 = opset18.Add(val_3, 1.0)
        val_7 = opset18.Mul(0.5, val_5)
        gelu = opset18.Mul(conv1d, val_7)
        conv1d_1 = opset18.Conv(
            gelu,
            encoder_encoder_conv2_weight,
            encoder_encoder_conv2_bias,
            group=1,
            pads=[1, 1],
            auto_pad="NOTSET",
            strides=[2],
            dilations=[1],
        )
        val_9 = opset18.Div(conv1d_1, 1.4142135)
        val_10 = opset18.Erf(val_9)
        val_12 = opset18.Add(val_10, 1.0)
        val_14 = opset18.Mul(0.5, val_12)
        gelu_1 = opset18.Mul(conv1d_1, val_14)
        permute = opset18.Transpose(gelu_1, perm=[0, 2, 1])
        add_20 = opset18.Add(permute, encoder_encoder_embed_positions_weight)
        layer_norm = opset18.LayerNormalization(
            add_20,
            encoder_encoder_layers_0_self_attn_layer_norm_weight,
            encoder_encoder_layers_0_self_attn_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_17 = opset18.Transpose(
            encoder_encoder_layers_0_self_attn_q_proj_weight, perm=[1, 0]
        )
        val_18 = opset18.MatMul(layer_norm, val_17)
        linear = opset18.Add(val_18, encoder_encoder_layers_0_self_attn_q_proj_bias)
        mul_18 = opset18.Mul(linear, 0.125)
        val_25 = opset18.Concat(val_0, [1500], [6], [64], axis=0)
        view = opset18.Reshape(mul_18, val_25, allowzero=0)
        transpose = opset18.Transpose(view, perm=[0, 2, 1, 3])
        val_27 = opset18.Transpose(
            encoder_encoder_layers_0_self_attn_k_proj_weight, perm=[1, 0]
        )
        linear_1 = opset18.MatMul(layer_norm, val_27)
        val_31 = opset18.Concat(val_0, [-1], [6], [64], axis=0)
        view_1 = opset18.Reshape(linear_1, val_31, allowzero=0)
        val_33 = opset18.Transpose(
            encoder_encoder_layers_0_self_attn_v_proj_weight, perm=[1, 0]
        )
        val_34 = opset18.MatMul(layer_norm, val_33)
        linear_2 = opset18.Add(val_34, encoder_encoder_layers_0_self_attn_v_proj_bias)
        val_37 = opset18.Concat(val_0, [-1], [6], [64], axis=0)
        view_2 = opset18.Reshape(linear_2, val_37, allowzero=0)
        transpose_2 = opset18.Transpose(view_2, perm=[0, 2, 1, 3])
        transpose_3 = opset18.Transpose(view_1, perm=[0, 2, 3, 1])
        matmul = opset18.MatMul(transpose, transpose_3)
        softmax = opset18.Softmax(matmul, axis=-1)
        matmul_1 = opset18.MatMul(softmax, transpose_2)
        transpose_4 = opset18.Transpose(matmul_1, perm=[0, 2, 1, 3])
        val_42 = opset18.Concat(val_0, [1500], [384], axis=0)
        _unsafe_view = opset18.Reshape(transpose_4, val_42, allowzero=0)
        val_44 = opset18.Transpose(
            encoder_encoder_layers_0_self_attn_out_proj_weight, perm=[1, 0]
        )
        val_45 = opset18.MatMul(_unsafe_view, val_44)
        linear_3 = opset18.Add(val_45, encoder_encoder_layers_0_self_attn_out_proj_bias)
        add_141 = opset18.Add(add_20, linear_3)
        layer_norm_1 = opset18.LayerNormalization(
            add_141,
            encoder_encoder_layers_0_final_layer_norm_weight,
            encoder_encoder_layers_0_final_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_48 = opset18.Transpose(encoder_encoder_layers_0_fc1_weight, perm=[1, 0])
        val_49 = opset18.MatMul(layer_norm_1, val_48)
        linear_4 = opset18.Add(val_49, encoder_encoder_layers_0_fc1_bias)
        val_51 = opset18.Div(linear_4, 1.4142135)
        val_52 = opset18.Erf(val_51)
        val_54 = opset18.Add(val_52, 1.0)
        val_56 = opset18.Mul(0.5, val_54)
        gelu_2 = opset18.Mul(linear_4, val_56)
        val_57 = opset18.Transpose(encoder_encoder_layers_0_fc2_weight, perm=[1, 0])
        val_58 = opset18.MatMul(gelu_2, val_57)
        linear_5 = opset18.Add(val_58, encoder_encoder_layers_0_fc2_bias)
        add_170 = opset18.Add(add_141, linear_5)
        layer_norm_2 = opset18.LayerNormalization(
            add_170,
            encoder_encoder_layer_norm_weight,
            encoder_encoder_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        return layer_norm_2

    model = main_graph.to_model_proto()
    return model


def make_model_with_random_weights():
    np.random.seed(10)  # Set a fixed seed
    encoder_encoder_embed_positions_weight = np.random.rand(1500, 384).astype(np.float32)
    encoder_encoder_conv1_weight = np.random.rand(384, 80, 3).astype(np.float32)
    encoder_encoder_conv1_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_conv2_weight = np.random.rand(384, 384, 3).astype(np.float32)
    encoder_encoder_conv2_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_self_attn_layer_norm_weight = np.random.rand(384).astype(
        np.float32
    )
    encoder_encoder_layers_0_self_attn_layer_norm_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_self_attn_q_proj_weight = np.random.rand(384, 384).astype(
        np.float32
    )
    encoder_encoder_layers_0_self_attn_q_proj_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_self_attn_k_proj_weight = np.random.rand(384, 384).astype(
        np.float32
    )
    encoder_encoder_layers_0_self_attn_v_proj_weight = np.random.rand(384, 384).astype(
        np.float32
    )
    encoder_encoder_layers_0_self_attn_v_proj_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_self_attn_out_proj_weight = np.random.rand(384, 384).astype(
        np.float32
    )
    encoder_encoder_layers_0_self_attn_out_proj_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_final_layer_norm_weight = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_final_layer_norm_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layers_0_fc1_weight = np.random.rand(1536, 384).astype(np.float32)
    encoder_encoder_layers_0_fc1_bias = np.random.rand(1536).astype(np.float32)
    encoder_encoder_layers_0_fc2_weight = np.random.rand(384, 1536).astype(np.float32)
    encoder_encoder_layers_0_fc2_bias = np.random.rand(384).astype(np.float32)
    encoder_encoder_layer_norm_weight = np.random.rand(384).astype(np.float32)
    encoder_encoder_layer_norm_bias = np.random.rand(384).astype(np.float32)
    model = make_model(
        encoder_encoder_embed_positions_weight,
        encoder_encoder_conv1_weight,
        encoder_encoder_conv1_bias,
        encoder_encoder_conv2_weight,
        encoder_encoder_conv2_bias,
        encoder_encoder_layers_0_self_attn_layer_norm_weight,
        encoder_encoder_layers_0_self_attn_layer_norm_bias,
        encoder_encoder_layers_0_self_attn_q_proj_weight,
        encoder_encoder_layers_0_self_attn_q_proj_bias,
        encoder_encoder_layers_0_self_attn_k_proj_weight,
        encoder_encoder_layers_0_self_attn_v_proj_weight,
        encoder_encoder_layers_0_self_attn_v_proj_bias,
        encoder_encoder_layers_0_self_attn_out_proj_weight,
        encoder_encoder_layers_0_self_attn_out_proj_bias,
        encoder_encoder_layers_0_final_layer_norm_weight,
        encoder_encoder_layers_0_final_layer_norm_bias,
        encoder_encoder_layers_0_fc1_weight,
        encoder_encoder_layers_0_fc1_bias,
        encoder_encoder_layers_0_fc2_weight,
        encoder_encoder_layers_0_fc2_bias,
        encoder_encoder_layer_norm_weight,
        encoder_encoder_layer_norm_bias,
    )
    return model


class _WhisperEncoderTest:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            np.random.seed(10)  # Set a fixed seed
            inputs = {
                "audio_features": np.random.rand(1, 80, 3000).astype(np.float32),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def whisper_encoder_test():
    return _WhisperEncoderTest()
