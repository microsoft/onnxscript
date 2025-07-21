# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Onnxscript version of "hf-internal-testing_tiny-random-bart".

See: https://huggingface.co/hf-internal-testing/tiny-random-bart
"""

import numpy as np

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset20
from onnxscript.onnx_types import FLOAT, INT64


def make_model(
    encoder_embed_tokens_weight,
    encoder_embed_positions_weight,
    encoder_layers_0_self_attn_k_proj_bias,
    encoder_layers_0_self_attn_layer_norm_weight,
    encoder_layers_0_fc1_bias,
    matmul_257,
    matmul_267,
    matmul_268,
    matmul_270,
    matmul_271,
    matmul_272,
    matmul_273,
    matmul_283,
    matmul_284,
    matmul_286,
    matmul_287,
    matmul_288,
):
    @script()
    def main_graph(input_ids: INT64[1, None]) -> FLOAT[None, None, 16]:
        encoder_layernorm_embedding_bias = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )
        encoder_layernorm_embedding_weight = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )

        encoder_layers_1_final_layer_norm_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_final_layer_norm_weight = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )

        encoder_layers_1_fc2_bias = opset20.Identity(encoder_layers_0_self_attn_k_proj_bias)
        encoder_layers_1_self_attn_layer_norm_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_self_attn_layer_norm_weight = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )
        encoder_layers_1_self_attn_out_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_self_attn_q_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_self_attn_v_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_self_attn_k_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_final_layer_norm_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_final_layer_norm_weight = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )
        encoder_layers_0_fc2_bias = opset20.Identity(encoder_layers_0_self_attn_k_proj_bias)
        encoder_layers_1_fc1_bias = opset20.Identity(encoder_layers_0_fc1_bias)
        encoder_layers_0_self_attn_out_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_self_attn_q_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_self_attn_v_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )

        encoder_shape_output_0 = opset20.Shape(input_ids)
        encoder_constant_output_0 = opset20.Constant(value=1)
        encoder_gather_output_0 = opset20.Gather(
            encoder_shape_output_0, encoder_constant_output_0
        )

        encoder_constant_1_output_0 = opset20.Constant(value=[-1])
        unsqueeze_43 = opset20.Constant(value=[0])
        encoder_unsqueeze_output_0 = opset20.Unsqueeze(encoder_gather_output_0, unsqueeze_43)
        encoder_concat_output_0 = opset20.Concat(
            encoder_constant_1_output_0, encoder_unsqueeze_output_0, axis=0
        )
        encoder_reshape_output_0 = opset20.Reshape(
            input_ids, encoder_concat_output_0, allowzero=0
        )
        encoder_embed_tokens_gather_output_0 = opset20.Gather(
            encoder_embed_tokens_weight, encoder_reshape_output_0
        )
        encoder_embed_tokens_constant_output_0 = opset20.Constant(value=[1.0])
        encoder_embed_tokens_mul_output_0 = opset20.Mul(
            encoder_embed_tokens_gather_output_0, encoder_embed_tokens_constant_output_0
        )
        encoder_embed_positions_shape_output_0 = opset20.Shape(input_ids)
        encoder_embed_positions_constant_output_0 = opset20.Constant(value=0)
        encoder_embed_positions_gather_output_0 = opset20.Gather(
            encoder_embed_positions_shape_output_0,
            encoder_embed_positions_constant_output_0,
            axis=0,
        )
        encoder_embed_positions_constant_1_output_0 = opset20.Constant(value=0)
        encoder_embed_positions_cast_output_0 = opset20.Cast(encoder_gather_output_0, to=7)
        encoder_embed_positions_constant_2_output_0 = opset20.Constant(value=1)
        encoder_embed_positions_range_output_0 = opset20.Range(
            encoder_embed_positions_constant_1_output_0,
            encoder_embed_positions_cast_output_0,
            encoder_embed_positions_constant_2_output_0,
        )
        encoder_embed_positions_constant_3_output_0 = opset20.Constant(value=[0])
        encoder_embed_positions_unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_embed_positions_gather_output_0,
            encoder_embed_positions_constant_3_output_0,
        )
        encoder_embed_positions_constant_4_output_0 = opset20.Constant(value=[-1])
        encoder_embed_positions_concat_output_0 = opset20.Concat(
            encoder_embed_positions_unsqueeze_output_0,
            encoder_embed_positions_constant_4_output_0,
            axis=0,
        )
        encoder_embed_positions_constant_5_output_0 = opset20.Constant(value=[-1])
        encoder_embed_positions_reshape_output_0 = opset20.Reshape(
            encoder_embed_positions_concat_output_0,
            encoder_embed_positions_constant_5_output_0,
        )
        encoder_embed_positions_shape_1_output_0 = opset20.Shape(
            encoder_embed_positions_reshape_output_0
        )
        encoder_embed_positions_constantofshape_output_0 = opset20.ConstantOfShape(
            encoder_embed_positions_shape_1_output_0,
            value=ir.tensor(np.array([1], dtype=np.int64)),
        )
        encoder_embed_positions_constant_6_output_0 = opset20.Constant(value=[-1])
        encoder_embed_positions_mul_output_0 = opset20.Mul(
            encoder_embed_positions_constantofshape_output_0,
            encoder_embed_positions_constant_6_output_0,
        )
        encoder_embed_positions_equal_output_0 = opset20.Equal(
            encoder_embed_positions_reshape_output_0, encoder_embed_positions_mul_output_0
        )
        encoder_embed_positions_where_output_0 = opset20.Where(
            encoder_embed_positions_equal_output_0,
            encoder_embed_positions_constantofshape_output_0,
            encoder_embed_positions_reshape_output_0,
        )
        encoder_embed_positions_expand_output_0 = opset20.Expand(
            encoder_embed_positions_range_output_0, encoder_embed_positions_where_output_0
        )
        encoder_embed_positions_constant_7_output_0 = opset20.Constant(value=2)
        encoder_embed_positions_add_output_0 = opset20.Add(
            encoder_embed_positions_expand_output_0,
            encoder_embed_positions_constant_7_output_0,
        )
        encoder_embed_positions_gather_1_output_0 = opset20.Gather(
            encoder_embed_positions_weight, encoder_embed_positions_add_output_0
        )
        encoder_cast_output_0 = opset20.Cast(encoder_embed_positions_gather_1_output_0, to=1)
        encoder_add_output_0 = opset20.Add(
            encoder_embed_tokens_mul_output_0, encoder_cast_output_0
        )
        encoder_layernorm_embedding_layernormalization_output_0 = opset20.LayerNormalization(
            encoder_add_output_0,
            encoder_layernorm_embedding_weight,
            encoder_layernorm_embedding_bias,
            axis=-1,
            epsilon=9.999999747378752e-06,
        )
        encoder_layers_0_self_attn_shape_output_0 = opset20.Shape(
            encoder_layernorm_embedding_layernormalization_output_0
        )
        encoder_layers_0_self_attn_constant_output_0 = opset20.Constant(value=0)
        encoder_layers_0_self_attn_gather_output_0 = opset20.Gather(
            encoder_layers_0_self_attn_shape_output_0,
            encoder_layers_0_self_attn_constant_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_shape_1_output_0 = opset20.Shape(
            encoder_layernorm_embedding_layernormalization_output_0
        )
        encoder_layers_0_self_attn_constant_1_output_0 = opset20.Constant(value=1)
        encoder_layers_0_self_attn_gather_1_output_0 = opset20.Gather(
            encoder_layers_0_self_attn_shape_1_output_0,
            encoder_layers_0_self_attn_constant_1_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_q_proj_matmul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_layernormalization_output_0, matmul_257
        )
        encoder_layers_0_self_attn_q_proj_add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_q_proj_bias,
            encoder_layers_0_self_attn_q_proj_matmul_output_0,
        )
        unsqueeze_88 = opset20.Constant(value=[0])
        encoder_layers_0_self_attn_unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_gather_output_0, unsqueeze_88
        )
        encoder_layers_0_self_attn_constant_2_output_0 = opset20.Constant(value=[-1])
        encoder_layers_0_self_attn_constant_3_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_constant_4_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_concat_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_unsqueeze_output_0,
            encoder_layers_0_self_attn_constant_2_output_0,
            encoder_layers_0_self_attn_constant_3_output_0,
            encoder_layers_0_self_attn_constant_4_output_0,
            axis=0,
        )
        unsqueeze_97 = opset20.Constant(value=[0])
        encoder_layers_0_self_attn_unsqueeze_1_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_gather_output_0, unsqueeze_97
        )
        encoder_layers_0_self_attn_constant_5_output_0 = opset20.Constant(value=[-1])
        encoder_layers_0_self_attn_constant_6_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_constant_7_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_concat_1_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_unsqueeze_1_output_0,
            encoder_layers_0_self_attn_constant_5_output_0,
            encoder_layers_0_self_attn_constant_6_output_0,
            encoder_layers_0_self_attn_constant_7_output_0,
            axis=0,
        )
        unsqueeze_106 = opset20.Constant(value=[0])
        encoder_layers_0_self_attn_unsqueeze_2_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_gather_output_0, unsqueeze_106
        )
        encoder_layers_0_self_attn_constant_8_output_0 = opset20.Constant(value=[-1])
        encoder_layers_0_self_attn_constant_9_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_constant_10_output_0 = opset20.Constant(value=[4])
        encoder_layers_0_self_attn_concat_2_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_unsqueeze_2_output_0,
            encoder_layers_0_self_attn_constant_8_output_0,
            encoder_layers_0_self_attn_constant_9_output_0,
            encoder_layers_0_self_attn_constant_10_output_0,
            axis=0,
        )

        encoder_layers_0_self_attn_reshape_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_q_proj_add_output_0,
            encoder_layers_0_self_attn_concat_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_transpose_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_reshape_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_0_self_attn_k_proj_matmul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_layernormalization_output_0, matmul_267
        )
        encoder_layers_0_self_attn_k_proj_add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_k_proj_bias,
            encoder_layers_0_self_attn_k_proj_matmul_output_0,
        )
        encoder_layers_0_self_attn_v_proj_matmul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_layernormalization_output_0, matmul_268
        )
        encoder_layers_0_self_attn_v_proj_add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_v_proj_bias,
            encoder_layers_0_self_attn_v_proj_matmul_output_0,
        )
        encoder_layers_0_self_attn_reshape_1_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_k_proj_add_output_0,
            encoder_layers_0_self_attn_concat_1_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_reshape_2_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_v_proj_add_output_0,
            encoder_layers_0_self_attn_concat_2_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_transpose_1_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_reshape_2_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_0_self_attn_shape_2_output_0 = opset20.Shape(
            encoder_layers_0_self_attn_transpose_output_0
        )
        encoder_layers_0_self_attn_constant_11_output_0 = opset20.Constant(value=[-1])
        encoder_layers_0_self_attn_constant_12_output_0 = opset20.Constant(
            value=[9223372036854775807]
        )
        encoder_layers_0_self_attn_slice_output_0 = opset20.Slice(
            encoder_layers_0_self_attn_shape_2_output_0,
            encoder_layers_0_self_attn_constant_11_output_0,
            encoder_layers_0_self_attn_constant_12_output_0,
        )
        encoder_layers_0_self_attn_cast_output_0 = opset20.Cast(
            encoder_layers_0_self_attn_slice_output_0, to=1
        )
        encoder_layers_0_self_attn_sqrt_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_cast_output_0
        )
        encoder_layers_0_self_attn_constant_13_output_0 = opset20.Constant(value=[1.0])
        encoder_layers_0_self_attn_div_output_0 = opset20.Div(
            encoder_layers_0_self_attn_constant_13_output_0,
            encoder_layers_0_self_attn_sqrt_output_0,
        )
        encoder_layers_0_self_attn_cast_1_output_0 = opset20.Cast(
            encoder_layers_0_self_attn_div_output_0, to=1
        )
        encoder_layers_0_self_attn_transpose_2_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_reshape_1_output_0, perm=[0, 2, 3, 1]
        )
        encoder_layers_0_self_attn_sqrt_1_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_cast_1_output_0
        )
        encoder_layers_0_self_attn_mul_output_0 = opset20.Mul(
            encoder_layers_0_self_attn_transpose_output_0,
            encoder_layers_0_self_attn_sqrt_1_output_0,
        )
        encoder_layers_0_self_attn_sqrt_2_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_cast_1_output_0
        )
        encoder_layers_0_self_attn_mul_1_output_0 = opset20.Mul(
            encoder_layers_0_self_attn_transpose_2_output_0,
            encoder_layers_0_self_attn_sqrt_2_output_0,
        )
        encoder_layers_0_self_attn_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_mul_output_0, encoder_layers_0_self_attn_mul_1_output_0
        )
        encoder_layers_0_self_attn_softmax_output_0 = opset20.Softmax(
            encoder_layers_0_self_attn_matmul_output_0, axis=-1
        )
        encoder_layers_0_self_attn_matmul_1_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_softmax_output_0,
            encoder_layers_0_self_attn_transpose_1_output_0,
        )
        encoder_layers_0_self_attn_transpose_3_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_matmul_1_output_0, perm=[0, 2, 1, 3]
        )
        unsqueeze_145 = opset20.Constant(value=[0])
        encoder_layers_0_self_attn_unsqueeze_3_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_gather_output_0, unsqueeze_145
        )
        unsqueeze_147 = opset20.Constant(value=[0])
        encoder_layers_0_self_attn_unsqueeze_4_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_gather_1_output_0, unsqueeze_147
        )
        encoder_layers_0_self_attn_constant_14_output_0 = opset20.Constant(value=[16])
        encoder_layers_0_self_attn_concat_3_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_unsqueeze_3_output_0,
            encoder_layers_0_self_attn_unsqueeze_4_output_0,
            encoder_layers_0_self_attn_constant_14_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_reshape_3_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_transpose_3_output_0,
            encoder_layers_0_self_attn_concat_3_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_out_proj_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_reshape_3_output_0, matmul_270
        )
        encoder_layers_0_self_attn_out_proj_add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_out_proj_bias,
            encoder_layers_0_self_attn_out_proj_matmul_output_0,
        )
        encoder_layers_0_add_output_0 = opset20.Add(
            encoder_layernorm_embedding_layernormalization_output_0,
            encoder_layers_0_self_attn_out_proj_add_output_0,
        )
        encoder_layers_0_self_attn_layer_norm_layernormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_0_add_output_0,
                encoder_layers_0_self_attn_layer_norm_weight,
                encoder_layernorm_embedding_bias,
                axis=-1,
                epsilon=9.999999747378752e-0,
            )
        )
        encoder_layers_0_fc1_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_layer_norm_layernormalization_output_0, matmul_271
        )
        encoder_layers_0_fc1_add_output_0 = opset20.Add(
            encoder_layers_0_fc1_bias, encoder_layers_0_fc1_matmul_output_0
        )
        encoder_layers_0_activation_fn_gelu_output_0 = opset20.Gelu(
            encoder_layers_0_fc1_add_output_0, approximate="none"
        )
        encoder_layers_0_fc2_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_activation_fn_gelu_output_0, matmul_272
        )
        encoder_layers_0_fc2_add_output_0 = opset20.Add(
            encoder_layers_0_fc2_bias, encoder_layers_0_fc2_matmul_output_0
        )
        encoder_layers_0_add_1_output_0 = opset20.Add(
            encoder_layers_0_self_attn_layer_norm_layernormalization_output_0,
            encoder_layers_0_fc2_add_output_0,
        )
        encoder_layers_0_final_layer_norm_layernormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_0_add_1_output_0,
                encoder_layers_0_final_layer_norm_weight,
                encoder_layers_0_final_layer_norm_bias,
                axis=-1,
                epsilon=9.999999747378752e-06,
            )
        )
        encoder_layers_1_self_attn_shape_output_0 = opset20.Shape(
            encoder_layers_0_final_layer_norm_layernormalization_output_0
        )
        encoder_layers_1_self_attn_constant_output_0 = opset20.Constant(value=0)
        encoder_layers_1_self_attn_gather_output_0 = opset20.Gather(
            encoder_layers_1_self_attn_shape_output_0,
            encoder_layers_1_self_attn_constant_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_shape_1_output_0 = opset20.Shape(
            encoder_layers_0_final_layer_norm_layernormalization_output_0
        )
        encoder_layers_1_self_attn_constant_1_output_0 = opset20.Constant(value=1)
        encoder_layers_1_self_attn_gather_1_output_0 = opset20.Gather(
            encoder_layers_1_self_attn_shape_1_output_0,
            encoder_layers_1_self_attn_constant_1_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_q_proj_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_layernormalization_output_0, matmul_273
        )
        encoder_layers_1_self_attn_q_proj_add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_q_proj_bias,
            encoder_layers_1_self_attn_q_proj_matmul_output_0,
        )
        unsqueeze_176 = opset20.Constant(value=[0])
        encoder_layers_1_self_attn_unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_gather_output_0, unsqueeze_176
        )
        encoder_layers_1_self_attn_constant_2_output_0 = opset20.Constant(value=[-1])
        encoder_layers_1_self_attn_constant_3_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_constant_4_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_concat_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_unsqueeze_output_0,
            encoder_layers_1_self_attn_constant_2_output_0,
            encoder_layers_1_self_attn_constant_3_output_0,
            encoder_layers_1_self_attn_constant_4_output_0,
            axis=0,
        )
        unsqueeze_185 = opset20.Constant(value=[0])
        encoder_layers_1_self_attn_unsqueeze_1_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_gather_output_0, unsqueeze_185
        )
        encoder_layers_1_self_attn_constant_5_output_0 = opset20.Constant(value=[-1])
        encoder_layers_1_self_attn_constant_6_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_constant_7_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_concat_1_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_unsqueeze_1_output_0,
            encoder_layers_1_self_attn_constant_5_output_0,
            encoder_layers_1_self_attn_constant_6_output_0,
            encoder_layers_1_self_attn_constant_7_output_0,
            axis=0,
        )
        unsqueeze_194 = opset20.Constant(value=[0])
        encoder_layers_1_self_attn_unsqueeze_2_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_gather_output_0, unsqueeze_194
        )
        encoder_layers_1_self_attn_constant_8_output_0 = opset20.Constant(value=[-1])
        encoder_layers_1_self_attn_constant_9_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_constant_10_output_0 = opset20.Constant(value=[4])
        encoder_layers_1_self_attn_concat_2_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_unsqueeze_2_output_0,
            encoder_layers_1_self_attn_constant_8_output_0,
            encoder_layers_1_self_attn_constant_9_output_0,
            encoder_layers_1_self_attn_constant_10_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_reshape_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_q_proj_add_output_0,
            encoder_layers_1_self_attn_concat_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_transpose_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_reshape_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_1_self_attn_k_proj_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_layernormalization_output_0, matmul_283
        )
        encoder_layers_1_self_attn_k_proj_add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_k_proj_bias,
            encoder_layers_1_self_attn_k_proj_matmul_output_0,
        )
        encoder_layers_1_self_attn_v_proj_matmul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_layernormalization_output_0, matmul_284
        )
        encoder_layers_1_self_attn_v_proj_add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_v_proj_bias,
            encoder_layers_1_self_attn_v_proj_matmul_output_0,
        )
        encoder_layers_1_self_attn_reshape_1_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_k_proj_add_output_0,
            encoder_layers_1_self_attn_concat_1_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_reshape_2_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_v_proj_add_output_0,
            encoder_layers_1_self_attn_concat_2_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_transpose_1_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_reshape_2_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_1_self_attn_shape_2_output_0 = opset20.Shape(
            encoder_layers_1_self_attn_transpose_output_0
        )
        encoder_layers_1_self_attn_constant_11_output_0 = opset20.Constant(value=[-1])
        encoder_layers_1_self_attn_constant_12_output_0 = opset20.Constant(
            value=[9223372036854775807]
        )
        encoder_layers_1_self_attn_slice_output_0 = opset20.Slice(
            encoder_layers_1_self_attn_shape_2_output_0,
            encoder_layers_1_self_attn_constant_11_output_0,
            encoder_layers_1_self_attn_constant_12_output_0,
        )
        encoder_layers_1_self_attn_cast_output_0 = opset20.Cast(
            encoder_layers_1_self_attn_slice_output_0, to=1
        )
        encoder_layers_1_self_attn_sqrt_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_cast_output_0
        )
        encoder_layers_1_self_attn_constant_13_output_0 = opset20.Constant(value=[1.0])
        encoder_layers_1_self_attn_div_output_0 = opset20.Div(
            encoder_layers_1_self_attn_constant_13_output_0,
            encoder_layers_1_self_attn_sqrt_output_0,
        )
        encoder_layers_1_self_attn_cast_1_output_0 = opset20.Cast(
            encoder_layers_1_self_attn_div_output_0, to=1
        )
        encoder_layers_1_self_attn_transpose_2_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_reshape_1_output_0, perm=[0, 2, 3, 1]
        )
        encoder_layers_1_self_attn_sqrt_1_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_cast_1_output_0
        )
        encoder_layers_1_self_attn_mul_output_0 = opset20.Mul(
            encoder_layers_1_self_attn_transpose_output_0,
            encoder_layers_1_self_attn_sqrt_1_output_0,
        )
        encoder_layers_1_self_attn_sqrt_2_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_cast_1_output_0
        )
        encoder_layers_1_self_attn_mul_1_output_0 = opset20.Mul(
            encoder_layers_1_self_attn_transpose_2_output_0,
            encoder_layers_1_self_attn_sqrt_2_output_0,
        )
        encoder_layers_1_self_attn_matmul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_mul_output_0, encoder_layers_1_self_attn_mul_1_output_0
        )
        encoder_layers_1_self_attn_softmax_output_0 = opset20.Softmax(
            encoder_layers_1_self_attn_matmul_output_0, axis=-1
        )
        encoder_layers_1_self_attn_matmul_1_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_softmax_output_0,
            encoder_layers_1_self_attn_transpose_1_output_0,
        )
        encoder_layers_1_self_attn_transpose_3_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_matmul_1_output_0, perm=[0, 2, 1, 3]
        )
        unsqueeze_232 = opset20.Constant(value=[0])
        encoder_layers_1_self_attn_unsqueeze_3_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_gather_output_0, unsqueeze_232
        )
        unsqueeze_234 = opset20.Constant(value=[0])
        encoder_layers_1_self_attn_unsqueeze_4_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_gather_1_output_0, unsqueeze_234
        )
        encoder_layers_1_self_attn_constant_14_output_0 = opset20.Constant(value=[16])

        encoder_layers_1_self_attn_concat_3_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_unsqueeze_3_output_0,
            encoder_layers_1_self_attn_unsqueeze_4_output_0,
            encoder_layers_1_self_attn_constant_14_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_reshape_3_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_transpose_3_output_0,
            encoder_layers_1_self_attn_concat_3_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_out_proj_matmul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_reshape_3_output_0, matmul_286
        )
        encoder_layers_1_self_attn_out_proj_add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_out_proj_bias,
            encoder_layers_1_self_attn_out_proj_matmul_output_0,
        )
        encoder_layers_1_add_output_0 = opset20.Add(
            encoder_layers_0_final_layer_norm_layernormalization_output_0,
            encoder_layers_1_self_attn_out_proj_add_output_0,
        )
        encoder_layers_1_self_attn_layer_norm_layernormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_1_add_output_0,
                encoder_layers_1_self_attn_layer_norm_weight,
                encoder_layers_1_self_attn_layer_norm_bias,
                axis=-1,
                epsilon=9.999999747378752e-06,
            )
        )
        encoder_layers_1_fc1_matmul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_layer_norm_layernormalization_output_0, matmul_287
        )
        encoder_layers_1_fc1_add_output_0 = opset20.Add(
            encoder_layers_1_fc1_bias, encoder_layers_1_fc1_matmul_output_0
        )
        encoder_layers_1_activation_fn_gelu_output_0 = opset20.Gelu(
            encoder_layers_1_fc1_add_output_0, approximate="none"
        )
        encoder_layers_1_fc2_matmul_output_0 = opset20.MatMul(
            encoder_layers_1_activation_fn_gelu_output_0, matmul_288
        )
        encoder_layers_1_fc2_add_output_0 = opset20.Add(
            encoder_layers_1_fc2_bias, encoder_layers_1_fc2_matmul_output_0
        )
        encoder_layers_1_add_1_output_0 = opset20.Add(
            encoder_layers_1_self_attn_layer_norm_layernormalization_output_0,
            encoder_layers_1_fc2_add_output_0,
        )
        encoder_output = opset20.LayerNormalization(
            encoder_layers_1_add_1_output_0,
            encoder_layers_1_final_layer_norm_weight,
            encoder_layers_1_final_layer_norm_bias,
            axis=-1,
            epsilon=9.999999747378752e-06,
        )
        return encoder_output

    return main_graph.to_model_proto()


def make_model_with_random_weights():
    encoder_embed_tokens_weight = np.random.rand(1024, 16).astype(np.float32)
    encoder_embed_positions_weight = np.random.rand(102, 16).astype(np.float32)
    encoder_layers_0_self_attn_k_proj_bias = np.random.rand(16).astype(np.float32)
    encoder_layers_0_self_attn_layer_norm_weight = np.random.rand(16).astype(np.float32)
    encoder_layers_0_fc1_bias = np.zeros((4), dtype=np.float32)

    matmul_257 = np.random.rand(16, 16).astype(np.float32)
    matmul_267 = np.random.rand(16, 16).astype(np.float32)
    matmul_268 = np.random.rand(16, 16).astype(np.float32)
    matmul_270 = np.random.rand(16, 16).astype(np.float32)
    matmul_271 = np.random.rand(16, 4).astype(np.float32)
    matmul_272 = np.random.rand(4, 16).astype(np.float32)
    matmul_273 = np.random.rand(16, 16).astype(np.float32)
    matmul_283 = np.random.rand(16, 16).astype(np.float32)
    matmul_284 = np.random.rand(16, 16).astype(np.float32)
    matmul_286 = np.random.rand(16, 16).astype(np.float32)
    matmul_287 = np.random.rand(16, 4).astype(np.float32)
    matmul_288 = np.random.rand(4, 16).astype(np.float32)

    model = make_model(
        encoder_embed_positions_weight=encoder_embed_positions_weight,
        encoder_embed_tokens_weight=encoder_embed_tokens_weight,
        encoder_layers_0_self_attn_k_proj_bias=encoder_layers_0_self_attn_k_proj_bias,
        encoder_layers_0_self_attn_layer_norm_weight=encoder_layers_0_self_attn_layer_norm_weight,
        encoder_layers_0_fc1_bias=encoder_layers_0_fc1_bias,
        matmul_257=matmul_257,
        matmul_267=matmul_267,
        matmul_268=matmul_268,
        matmul_270=matmul_270,
        matmul_271=matmul_271,
        matmul_272=matmul_272,
        matmul_273=matmul_273,
        matmul_283=matmul_283,
        matmul_284=matmul_284,
        matmul_286=matmul_286,
        matmul_287=matmul_287,
        matmul_288=matmul_288,
    )
    return model


class _BartEncoderTest:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "input_ids": np.random.randint(0, 1024, (1, 16)).astype(np.int64),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def bart_encoder_test():
    return _BartEncoderTest()
