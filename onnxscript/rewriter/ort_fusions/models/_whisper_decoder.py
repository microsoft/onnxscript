# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A one-layer Whisper decoder model test case, with inputs: audio_features.
This model contains one layer of self-attention and one layer of cross-attention.
This is an onnxscript version of the model.
"""

import numpy as np

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT, INT32


def make_model(
    decoder_embed_positions_weight,
    proj_out_weight,
    decoder_layers_0_self_attn_layer_norm_weight,
    decoder_layers_0_self_attn_layer_norm_bias,
    decoder_layers_0_self_attn_q_proj_weight,
    decoder_layers_0_self_attn_q_proj_bias,
    decoder_layers_0_self_attn_k_proj_weight,
    decoder_layers_0_self_attn_v_proj_weight,
    decoder_layers_0_self_attn_v_proj_bias,
    decoder_layers_0_self_attn_out_proj_weight,
    decoder_layers_0_self_attn_out_proj_bias,
    decoder_layers_0_encoder_attn_layer_norm_weight,
    decoder_layers_0_encoder_attn_layer_norm_bias,
    decoder_layers_0_encoder_attn_q_proj_weight,
    decoder_layers_0_encoder_attn_q_proj_bias,
    decoder_layers_0_encoder_attn_out_proj_weight,
    decoder_layers_0_encoder_attn_out_proj_bias,
    decoder_layers_0_final_layer_norm_weight,
    decoder_layers_0_final_layer_norm_bias,
    decoder_layers_0_fc1_weight,
    decoder_layers_0_fc1_bias,
    decoder_layers_0_fc2_weight,
    decoder_layers_0_fc2_bias,
    decoder_layer_norm_weight,
    decoder_layer_norm_bias,
):
    @script()
    def main_graph(
        # TODO: Fix test case for dynamic batch size and past sequence length
        decoder_input_ids: INT32[1, 1],
        encoder_hidden_states: FLOAT[1, 1500, 384],
        past_key_values_0_0: FLOAT[1, 6, 32, 64],
        past_key_values_0_1: FLOAT[1, 6, 32, 64],
        past_key_values_0_2: FLOAT[1, 6, 32, 64],
        past_key_values_0_3: FLOAT[1, 6, 32, 64],
    ) -> (
        FLOAT[1, 1, 51865],
        FLOAT[1, 6, 33, 64],
        FLOAT[1, 6, 33, 64],
    ):
        val_0 = opset18.Shape(decoder_input_ids, end=1, start=0)
        val_1 = opset18.Shape(past_key_values_0_0, end=3, start=2)
        sym_size_int_42 = opset18.Squeeze(val_1)
        view = opset18.Reshape(decoder_input_ids, [-1, 1], allowzero=0)
        embedding = opset18.Gather(proj_out_weight, view, axis=0)
        add_7 = opset18.Add(sym_size_int_42, 1)
        arange = opset18.Range(sym_size_int_42, add_7, 1)
        unsqueeze = opset18.Unsqueeze(arange, [0])
        val_16 = opset18.Concat(val_0, [1], axis=0)
        repeat = opset18.Tile(unsqueeze, val_16)
        val_22 = opset18.Unsqueeze(repeat, [-1])
        val_24 = opset18.GatherND(decoder_embed_positions_weight, val_22, batch_dims=0)
        add_15 = opset18.Add(embedding, val_24)
        add_24 = opset18.Add(add_7, 1)
        val_28 = opset18.Reshape(add_24, [-1], allowzero=0)
        val_29 = opset18.Concat([1], val_28, axis=0)
        full = opset18.Expand(-3.4028235e38, val_29)
        arange_1 = opset18.Range(0, add_24, 1)
        view_1 = opset18.Reshape(arange, [-1, 1], allowzero=0)
        gt = opset18.Greater(arange_1, view_1)
        convert_element_type_default = opset18.Cast(gt, to=1)
        mul_17 = opset18.Mul(full, convert_element_type_default)
        layer_norm = opset18.LayerNormalization(
            add_15,
            decoder_layers_0_self_attn_layer_norm_weight,
            decoder_layers_0_self_attn_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_37 = opset18.Transpose(decoder_layers_0_self_attn_q_proj_weight, perm=[1, 0])
        val_38 = opset18.MatMul(layer_norm, val_37)
        linear = opset18.Add(val_38, decoder_layers_0_self_attn_q_proj_bias)
        mul_43 = opset18.Mul(linear, 0.125)
        val_44 = opset18.Concat(val_0, [1], [6], [64], axis=0)
        view_2 = opset18.Reshape(mul_43, val_44, allowzero=0)
        transpose = opset18.Transpose(view_2, perm=[0, 2, 1, 3])
        val_46 = opset18.Transpose(decoder_layers_0_self_attn_k_proj_weight, perm=[1, 0])
        linear_1 = opset18.MatMul(layer_norm, val_46)
        val_49 = opset18.Concat(val_0, [-1], [6], [64], axis=0)
        view_3 = opset18.Reshape(linear_1, val_49, allowzero=0)
        transpose_1 = opset18.Transpose(view_3, perm=[0, 2, 1, 3])
        val_51 = opset18.Transpose(decoder_layers_0_self_attn_v_proj_weight, perm=[1, 0])
        val_52 = opset18.MatMul(layer_norm, val_51)
        linear_2 = opset18.Add(val_52, decoder_layers_0_self_attn_v_proj_bias)
        val_55 = opset18.Concat(val_0, [-1], [6], [64], axis=0)
        view_4 = opset18.Reshape(linear_2, val_55, allowzero=0)
        transpose_2 = opset18.Transpose(view_4, perm=[0, 2, 1, 3])
        cat = opset18.Concat(past_key_values_0_0, transpose_1, axis=-2)
        cat_1 = opset18.Concat(past_key_values_0_1, transpose_2, axis=-2)
        transpose_3 = opset18.Transpose(cat, perm=[0, 1, 3, 2])
        matmul = opset18.MatMul(transpose, transpose_3)
        unsqueeze_4 = opset18.Unsqueeze(mul_17, [0, 1])
        val_83 = opset18.Concat(val_0, [1], [-1], [-1], axis=0)
        val_85 = opset18.Abs(val_83)
        expand_1 = opset18.Expand(unsqueeze_4, val_85)
        val_104 = opset18.Constant(value_ints=[0])
        val_106 = opset18.Constant(value_ints=[-1])
        val_107 = opset18.Reshape(add_7, val_106, allowzero=0)
        val_111 = opset18.Constant(value_ints=[1])
        slice_12 = opset18.Slice(expand_1, val_104, val_107, [3], val_111)
        add_125 = opset18.Add(matmul, slice_12)
        softmax = opset18.Softmax(add_125, axis=-1)
        matmul_1 = opset18.MatMul(softmax, cat_1)
        transpose_4 = opset18.Transpose(matmul_1, perm=[0, 2, 1, 3])
        val_115 = opset18.Concat(val_0, [1], [384], axis=0)
        view_5 = opset18.Reshape(transpose_4, val_115, allowzero=0)
        val_117 = opset18.Transpose(decoder_layers_0_self_attn_out_proj_weight, perm=[1, 0])
        val_118 = opset18.MatMul(view_5, val_117)
        linear_3 = opset18.Add(val_118, decoder_layers_0_self_attn_out_proj_bias)
        add_163 = opset18.Add(add_15, linear_3)
        layer_norm_1 = opset18.LayerNormalization(
            add_163,
            decoder_layers_0_encoder_attn_layer_norm_weight,
            decoder_layers_0_encoder_attn_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_121 = opset18.Transpose(decoder_layers_0_encoder_attn_q_proj_weight, perm=[1, 0])
        val_122 = opset18.MatMul(layer_norm_1, val_121)
        linear_4 = opset18.Add(val_122, decoder_layers_0_encoder_attn_q_proj_bias)
        mul_125 = opset18.Mul(linear_4, 0.125)
        val_125 = opset18.Concat(val_0, [1], [6], [64], axis=0)
        view_6 = opset18.Reshape(mul_125, val_125, allowzero=0)
        transpose_5 = opset18.Transpose(view_6, perm=[0, 2, 1, 3])
        transpose_6 = opset18.Transpose(past_key_values_0_2, perm=[0, 1, 3, 2])
        matmul_2 = opset18.MatMul(transpose_5, transpose_6)
        softmax_1 = opset18.Softmax(matmul_2, axis=-1)
        matmul_3 = opset18.MatMul(softmax_1, past_key_values_0_3)
        transpose_7 = opset18.Transpose(matmul_3, perm=[0, 2, 1, 3])
        val_129 = opset18.Concat(val_0, [1], [384], axis=0)
        view_7 = opset18.Reshape(transpose_7, val_129, allowzero=0)
        val_131 = opset18.Transpose(decoder_layers_0_encoder_attn_out_proj_weight, perm=[1, 0])
        val_132 = opset18.MatMul(view_7, val_131)
        linear_5 = opset18.Add(val_132, decoder_layers_0_encoder_attn_out_proj_bias)
        add_232 = opset18.Add(add_163, linear_5)
        layer_norm_2 = opset18.LayerNormalization(
            add_232,
            decoder_layers_0_final_layer_norm_weight,
            decoder_layers_0_final_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_135 = opset18.Transpose(decoder_layers_0_fc1_weight, perm=[1, 0])
        val_136 = opset18.MatMul(layer_norm_2, val_135)
        linear_6 = opset18.Add(val_136, decoder_layers_0_fc1_bias)
        val_138 = opset18.Div(linear_6, 1.4142135)
        val_139 = opset18.Erf(val_138)
        val_141 = opset18.Add(val_139, 1.0)
        val_143 = opset18.Mul(0.5, val_141)
        gelu = opset18.Mul(linear_6, val_143)
        val_144 = opset18.Transpose(decoder_layers_0_fc2_weight, perm=[1, 0])
        val_145 = opset18.MatMul(gelu, val_144)
        linear_7 = opset18.Add(val_145, decoder_layers_0_fc2_bias)
        add_261 = opset18.Add(add_232, linear_7)
        layer_norm_12 = opset18.LayerNormalization(
            add_261,
            decoder_layer_norm_weight,
            decoder_layer_norm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_457 = opset18.Transpose(proj_out_weight, perm=[1, 0])
        linear_32 = opset18.MatMul(layer_norm_12, val_457)
        return linear_32, cat, cat_1

    model = main_graph.to_model_proto()
    return model


def make_model_with_random_weights():
    np.random.seed(10)  # Set a fixed seed
    decoder_embed_positions_weight = np.random.rand(448, 384).astype(np.float32)
    proj_out_weight = np.random.rand(51865, 384).astype(np.float32)
    decoder_layers_0_self_attn_layer_norm_weight = np.random.rand(384).astype(np.float32)
    decoder_layers_0_self_attn_layer_norm_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_self_attn_q_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_self_attn_q_proj_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_self_attn_k_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_self_attn_v_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_self_attn_v_proj_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_self_attn_out_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_self_attn_out_proj_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_encoder_attn_layer_norm_weight = np.random.rand(384).astype(np.float32)
    decoder_layers_0_encoder_attn_layer_norm_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_encoder_attn_q_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_encoder_attn_q_proj_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_encoder_attn_out_proj_weight = np.random.rand(384, 384).astype(np.float32)
    decoder_layers_0_encoder_attn_out_proj_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_final_layer_norm_weight = np.random.rand(384).astype(np.float32)
    decoder_layers_0_final_layer_norm_bias = np.random.rand(384).astype(np.float32)
    decoder_layers_0_fc1_weight = np.random.rand(1536, 384).astype(np.float32)
    decoder_layers_0_fc1_bias = np.random.rand(1536).astype(np.float32)
    decoder_layers_0_fc2_weight = np.random.rand(384, 1536).astype(np.float32)
    decoder_layers_0_fc2_bias = np.random.rand(384).astype(np.float32)
    decoder_layer_norm_weight = np.random.rand(384).astype(np.float32)
    decoder_layer_norm_bias = np.random.rand(384).astype(np.float32)

    model = make_model(
        decoder_embed_positions_weight,
        proj_out_weight,
        decoder_layers_0_self_attn_layer_norm_weight,
        decoder_layers_0_self_attn_layer_norm_bias,
        decoder_layers_0_self_attn_q_proj_weight,
        decoder_layers_0_self_attn_q_proj_bias,
        decoder_layers_0_self_attn_k_proj_weight,
        decoder_layers_0_self_attn_v_proj_weight,
        decoder_layers_0_self_attn_v_proj_bias,
        decoder_layers_0_self_attn_out_proj_weight,
        decoder_layers_0_self_attn_out_proj_bias,
        decoder_layers_0_encoder_attn_layer_norm_weight,
        decoder_layers_0_encoder_attn_layer_norm_bias,
        decoder_layers_0_encoder_attn_q_proj_weight,
        decoder_layers_0_encoder_attn_q_proj_bias,
        decoder_layers_0_encoder_attn_out_proj_weight,
        decoder_layers_0_encoder_attn_out_proj_bias,
        decoder_layers_0_final_layer_norm_weight,
        decoder_layers_0_final_layer_norm_bias,
        decoder_layers_0_fc1_weight,
        decoder_layers_0_fc1_bias,
        decoder_layers_0_fc2_weight,
        decoder_layers_0_fc2_bias,
        decoder_layer_norm_weight,
        decoder_layer_norm_bias,
    )
    return model


class _WhisperDecoderTest:
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
                "decoder_input_ids": np.random.randint(0, 49152, (1, 1)).astype(np.int32),
                "encoder_hidden_states": np.random.rand(1, 1500, 384).astype(np.float32),
                "past_key_values_0_0": np.random.rand(1, 6, 32, 64).astype(np.float32),
                "past_key_values_0_1": np.random.rand(1, 6, 32, 64).astype(np.float32),
                "past_key_values_0_2": np.random.rand(1, 6, 32, 64).astype(np.float32),
                "past_key_values_0_3": np.random.rand(1, 6, 32, 64).astype(np.float32),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def whisper_decoder_test():
    return _WhisperDecoderTest()
