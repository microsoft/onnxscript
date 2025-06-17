"""
Onnxscript version of "hf-internal-testing_tiny-random-bart".

See: https://huggingface.co/hf-internal-testing/tiny-random-bart
"""

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor

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
    MatMul_257,
    MatMul_267,
    MatMul_268,
    MatMul_270,
    MatMul_271,
    MatMul_272,
    MatMul_273,
    MatMul_283,
    MatMul_284,
    MatMul_286,
    MatMul_287,
    MatMul_288,
):
    @script()
    def main_graph(input_ids: INT64[1, 16]) -> FLOAT[None]:
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
        encoder_layers_1_fc1_bias = opset20.Identity(encoder_layers_0_fc1_bias)
        encoder_layers_1_self_attn_layer_norm_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_1_self_attn_layer_norm_weight = opset20.Identity(
            encoder_layers_0_self_attn_layer_norm_weight
        )
        encoder_layers_1_self_attn_layer_norm_weight = opset20.Identity(
            encoder_layers_1_self_attn_layer_norm_weight
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
        encoder_layers_1_fc2_bias = opset20.Identity(encoder_layers_0_self_attn_k_proj_bias)
        encoder_layers_1_self_attn_layer_norm_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_self_attn_out_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_self_attn_q_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )
        encoder_layers_0_self_attn_v_proj_bias = opset20.Identity(
            encoder_layers_0_self_attn_k_proj_bias
        )

        encoder_Shape_output_0 = opset20.Shape(input_ids)
        encoder_Constant_output_0 = opset20.Constant(value_ints=1)
        encoder_Gather_output_0 = opset20.Gather(
            encoder_Shape_output_0, encoder_Constant_output_0
        )

        encoder_Constant_1_output_0 = opset20.Constant(value_int=-1)
        Unsqueeze_43 = opset20.Constant(int_values=[0])
        encoder_Unsqueeze_output_0 = opset20.Unsqueeze(encoder_Gather_output_0, Unsqueeze_43)
        encoder_Concat_output_0 = opset20.Concat(
            encoder_Constant_1_output_0, encoder_Unsqueeze_output_0, axis=0
        )
        encoder_Reshape_output_0 = opset20.Reshape(
            input_ids, encoder_Concat_output_0, allowzero=0
        )
        encoder_embed_tokens_Gather_output_0 = opset20.Gather(
            encoder_embed_tokens_weight, encoder_Reshape_output_0
        )
        encoder_embed_tokens_Constant_output_0 = opset20.Constant(value_floats=[1.0])
        encoder_embed_tokens_Mul_output_0 = opset20.Mul(
            encoder_embed_tokens_Gather_output_0, encoder_embed_tokens_Constant_output_0
        )
        encoder_embed_positions_Shape_output_0 = opset20.Shape(input_ids)
        encoder_embed_positions_Constant_output_0 = opset20.Constant(value_int=0)
        encoder_embed_positions_Gather_output_0 = opset20.Gather(
            encoder_embed_positions_Shape_output_0,
            encoder_embed_positions_Constant_output_0,
            axis=0,
        )
        encoder_embed_positions_Constant_1_output_0 = opset20.Constant(value_int=0)
        encoder_embed_positions_Cast_output_0 = opset20.Cast(encoder_Gather_output_0, to=7)
        encoder_embed_positions_Constant_2_output_0 = opset20.Constant(value_int=1)
        encoder_embed_positions_Range_output_0 = opset20.Range(
            encoder_embed_positions_Constant_1_output_0,
            encoder_embed_positions_Cast_output_0,
            encoder_embed_positions_Constant_2_output_0,
        )
        encoder_embed_positions_Constant_3_output_0 = opset20.Constant(value_ints=[0])
        encoder_embed_positions_Unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_embed_positions_Gather_output_0,
            encoder_embed_positions_Constant_3_output_0,
        )
        encoder_embed_positions_Constant_4_output_0 = opset20.Constant(value_ints=[-1])
        encoder_embed_positions_Concat_output_0 = opset20.Concat(
            encoder_embed_positions_Unsqueeze_output_0,
            encoder_embed_positions_Constant_4_output_0,
            axis=0,
        )
        encoder_embed_positions_Constant_5_output_0 = opset20.Constant(value_ints=[-1])
        encoder_embed_positions_Reshape_output_0 = opset20.Reshape(
            encoder_embed_positions_Concat_output_0,
            encoder_embed_positions_Constant_5_output_0,
        )
        encoder_embed_positions_Shape_1_output_0 = opset20.Shape(
            encoder_embed_positions_Reshape_output_0
        )
        encoder_embed_positions_ConstantOfShape_output_0 = opset20.ConstantOfShape(
            encoder_embed_positions_Shape_1_output_0,
            value=make_tensor("onef", TensorProto.FLOAT, [1], [1]),
        )
        encoder_embed_positions_Constant_6_output_0 = opset20.Constant(value_ints=[-1])
        encoder_embed_positions_Mul_output_0 = opset20.Mul(
            encoder_embed_positions_ConstantOfShape_output_0,
            encoder_embed_positions_Constant_6_output_0,
        )
        encoder_embed_positions_Equal_output_0 = opset20.Equal(
            encoder_embed_positions_Reshape_output_0, encoder_embed_positions_Mul_output_0
        )
        encoder_embed_positions_Where_output_0 = opset20.Where(
            encoder_embed_positions_Equal_output_0,
            encoder_embed_positions_ConstantOfShape_output_0,
            encoder_embed_positions_Reshape_output_0,
        )
        encoder_embed_positions_Expand_output_0 = opset20.Expand(
            encoder_embed_positions_Range_output_0, encoder_embed_positions_Where_output_0
        )
        encoder_embed_positions_Constant_7_output_0 = opset20.Constant(value_int=2)
        encoder_embed_positions_Add_output_0 = opset20.Add(
            encoder_embed_positions_Expand_output_0,
            encoder_embed_positions_Constant_7_output_0,
        )
        encoder_embed_positions_Gather_1_output_0 = opset20.Gather(
            encoder_embed_positions_weight, encoder_embed_positions_Add_output_0
        )
        encoder_Cast_output_0 = opset20.Cast(encoder_embed_positions_Gather_1_output_0, to=1)
        encoder_Add_output_0 = opset20.Add(
            encoder_embed_tokens_Mul_output_0, encoder_Cast_output_0
        )
        encoder_layernorm_embedding_LayerNormalization_output_0 = opset20.LayerNormalization(
            encoder_Add_output_0,
            encoder_layernorm_embedding_weight,
            encoder_layernorm_embedding_bias,
            axis=-1,
            epsilon=9.999999747378752e-06,
        )
        encoder_layers_0_self_attn_Shape_output_0 = opset20.Shape(
            encoder_layernorm_embedding_LayerNormalization_output_0
        )
        encoder_layers_0_self_attn_Constant_output_0 = opset20.Constant(value_int=0)
        encoder_layers_0_self_attn_Gather_output_0 = opset20.Gather(
            encoder_layers_0_self_attn_Shape_output_0,
            encoder_layers_0_self_attn_Constant_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_Shape_1_output_0 = opset20.Shape(
            encoder_layernorm_embedding_LayerNormalization_output_0
        )
        encoder_layers_0_self_attn_Constant_1_output_0 = opset20.Constant(value_int=1)
        encoder_layers_0_self_attn_Gather_1_output_0 = opset20.Gather(
            encoder_layers_0_self_attn_Shape_1_output_0,
            encoder_layers_0_self_attn_Constant_1_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_q_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_LayerNormalization_output_0, MatMul_257
        )
        encoder_layers_0_self_attn_q_proj_Add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_q_proj_bias,
            encoder_layers_0_self_attn_q_proj_MatMul_output_0,
        )
        Unsqueeze_88 = opset20.Constant(value_ints=[0])
        encoder_layers_0_self_attn_Unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_Gather_output_0, Unsqueeze_88
        )
        encoder_layers_0_self_attn_Constant_2_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_0_self_attn_Constant_3_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Constant_4_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Concat_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_Unsqueeze_output_0,
            encoder_layers_0_self_attn_Constant_2_output_0,
            encoder_layers_0_self_attn_Constant_3_output_0,
            encoder_layers_0_self_attn_Constant_4_output_0,
            axis=0,
        )
        Unsqueeze_97 = opset20.Constant(value_ints=[1])
        encoder_layers_0_self_attn_Unsqueeze_1_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_Gather_output_0, Unsqueeze_97
        )
        encoder_layers_0_self_attn_Constant_5_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_0_self_attn_Constant_6_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Constant_7_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Concat_1_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_Unsqueeze_1_output_0,
            encoder_layers_0_self_attn_Constant_5_output_0,
            encoder_layers_0_self_attn_Constant_6_output_0,
            encoder_layers_0_self_attn_Constant_7_output_0,
            axis=0,
        )
        Unsqueeze_106 = opset20.Constant(value_ints=[1])
        encoder_layers_0_self_attn_Unsqueeze_2_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_Gather_output_0, Unsqueeze_106
        )
        encoder_layers_0_self_attn_Constant_8_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_0_self_attn_Constant_9_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Constant_10_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_0_self_attn_Concat_2_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_Unsqueeze_2_output_0,
            encoder_layers_0_self_attn_Constant_8_output_0,
            encoder_layers_0_self_attn_Constant_9_output_0,
            encoder_layers_0_self_attn_Constant_10_output_0,
            axis=0,
        )

        encoder_layers_0_self_attn_Reshape_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_q_proj_Add_output_0,
            encoder_layers_0_self_attn_Concat_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_Transpose_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_Reshape_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_0_self_attn_k_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_LayerNormalization_output_0, MatMul_267
        )
        encoder_layers_0_self_attn_k_proj_Add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_k_proj_bias,
            encoder_layers_0_self_attn_k_proj_MatMul_output_0,
        )
        encoder_layers_0_self_attn_v_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layernorm_embedding_LayerNormalization_output_0, MatMul_268
        )
        encoder_layers_0_self_attn_v_proj_Add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_v_proj_bias,
            encoder_layers_0_self_attn_v_proj_MatMul_output_0,
        )
        encoder_layers_0_self_attn_Reshape_1_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_k_proj_Add_output_0,
            encoder_layers_0_self_attn_Concat_1_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_Reshape_2_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_v_proj_Add_output_0,
            encoder_layers_0_self_attn_Concat_2_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_Transpose_1_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_Reshape_2_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_0_self_attn_Shape_2_output_0 = opset20.Shape(
            encoder_layers_0_self_attn_Transpose_output_0
        )
        encoder_layers_0_self_attn_Constant_11_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_0_self_attn_Constant_12_output_0 = opset20.Constant(
            value_ints=[9223372036854775807]
        )
        encoder_layers_0_self_attn_Slice_output_0 = opset20.Slice(
            encoder_layers_0_self_attn_Shape_2_output_0,
            encoder_layers_0_self_attn_Constant_11_output_0,
            encoder_layers_0_self_attn_Constant_12_output_0,
        )
        encoder_layers_0_self_attn_Cast_output_0 = opset20.Cast(
            encoder_layers_0_self_attn_Slice_output_0, to=1
        )
        encoder_layers_0_self_attn_Sqrt_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_Cast_output_0
        )
        encoder_layers_0_self_attn_Constant_13_output_0 = opset20.Constant(value_floats=[1.0])
        encoder_layers_0_self_attn_Div_output_0 = opset20.Div(
            encoder_layers_0_self_attn_Constant_13_output_0,
            encoder_layers_0_self_attn_Sqrt_output_0,
        )
        encoder_layers_0_self_attn_Cast_1_output_0 = opset20.Cast(
            encoder_layers_0_self_attn_Div_output_0, to=1
        )
        encoder_layers_0_self_attn_Transpose_2_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_Reshape_1_output_0, perm=[0, 2, 3, 1]
        )
        encoder_layers_0_self_attn_Sqrt_1_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_Cast_1_output_0
        )
        encoder_layers_0_self_attn_Mul_output_0 = opset20.Mul(
            encoder_layers_0_self_attn_Transpose_output_0,
            encoder_layers_0_self_attn_Sqrt_1_output_0,
        )
        encoder_layers_0_self_attn_Sqrt_2_output_0 = opset20.Sqrt(
            encoder_layers_0_self_attn_Cast_1_output_0
        )
        encoder_layers_0_self_attn_Mul_1_output_0 = opset20.Mul(
            encoder_layers_0_self_attn_Transpose_2_output_0,
            encoder_layers_0_self_attn_Sqrt_2_output_0,
        )
        encoder_layers_0_self_attn_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_Mul_output_0, encoder_layers_0_self_attn_Mul_1_output_0
        )
        encoder_layers_0_self_attn_Softmax_output_0 = opset20.Softmax(
            encoder_layers_0_self_attn_MatMul_output_0, axis=-1
        )
        encoder_layers_0_self_attn_MatMul_1_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_Softmax_output_0,
            encoder_layers_0_self_attn_Transpose_1_output_0,
        )
        encoder_layers_0_self_attn_Transpose_3_output_0 = opset20.Transpose(
            encoder_layers_0_self_attn_MatMul_1_output_0, perm=[0, 2, 1, 3]
        )
        Unsqueeze_145 = opset20.Constant(value_ints=[0])
        encoder_layers_0_self_attn_Unsqueeze_3_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_Gather_output_0, Unsqueeze_145
        )
        Unsqueeze_147 = opset20.Constant(value_ints=[0])
        encoder_layers_0_self_attn_Unsqueeze_4_output_0 = opset20.Unsqueeze(
            encoder_layers_0_self_attn_Gather_1_output_0, Unsqueeze_147
        )
        encoder_layers_0_self_attn_Constant_14_output_0 = opset20.Constant(value_ints=[16])
        encoder_layers_0_self_attn_Concat_3_output_0 = opset20.Concat(
            encoder_layers_0_self_attn_Unsqueeze_3_output_0,
            encoder_layers_0_self_attn_Unsqueeze_4_output_0,
            encoder_layers_0_self_attn_Constant_14_output_0,
            axis=0,
        )
        encoder_layers_0_self_attn_Reshape_3_output_0 = opset20.Reshape(
            encoder_layers_0_self_attn_Transpose_3_output_0,
            encoder_layers_0_self_attn_Concat_3_output_0,
            allowzero=0,
        )
        encoder_layers_0_self_attn_out_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_Reshape_3_output_0, MatMul_270
        )
        encoder_layers_0_self_attn_out_proj_Add_output_0 = opset20.Add(
            encoder_layers_0_self_attn_out_proj_bias,
            encoder_layers_0_self_attn_out_proj_MatMul_output_0,
        )
        encoder_layers_0_Add_output_0 = opset20.Add(
            encoder_layernorm_embedding_LayerNormalization_output_0,
            encoder_layers_0_self_attn_out_proj_Add_output_0,
        )
        encoder_layers_0_self_attn_layer_norm_LayerNormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_0_Add_output_0,
                encoder_layers_0_self_attn_layer_norm_weight,
                axis=-1,
                epsilon=9.999999747378752e-0,
            )
        )
        encoder_layers_0_fc1_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_self_attn_layer_norm_LayerNormalization_output_0, MatMul_271
        )
        encoder_layers_0_fc1_Add_output_0 = opset20.Add(
            encoder_layers_0_fc1_bias, encoder_layers_0_fc1_MatMul_output_0
        )
        encoder_layers_0_activation_fn_Gelu_output_0 = opset20.Gelu(
            encoder_layers_0_fc1_Add_output_0, approximate="none"
        )
        encoder_layers_0_fc2_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_activation_fn_Gelu_output_0, MatMul_272
        )
        encoder_layers_0_fc2_Add_output_0 = opset20.Add(
            encoder_layers_0_fc2_bias, encoder_layers_0_fc2_MatMul_output_0
        )
        encoder_layers_0_Add_1_output_0 = opset20.Add(
            encoder_layers_0_self_attn_layer_norm_LayerNormalization_output_0,
            encoder_layers_0_fc2_Add_output_0,
        )
        encoder_layers_0_final_layer_norm_LayerNormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_0_Add_1_output_0,
                encoder_layers_0_final_layer_norm_weight,
                encoder_layers_0_final_layer_norm_bias,
                axis=-1,
                epsilon=9.999999747378752e-06,
            )
        )
        encoder_layers_1_self_attn_Shape_output_0 = opset20.Shape(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0
        )
        encoder_layers_1_self_attn_Constant_output_0 = opset20.Constant(value_int=0)
        encoder_layers_1_self_attn_Gather_output_0 = opset20.Gather(
            encoder_layers_1_self_attn_Shape_output_0,
            encoder_layers_1_self_attn_Constant_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_Shape_1_output_0 = opset20.Shape(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0
        )
        encoder_layers_1_self_attn_Constant_1_output_0 = opset20.Constant(value_int=1)
        encoder_layers_1_self_attn_Gather_1_output_0 = opset20.Gather(
            encoder_layers_1_self_attn_Shape_1_output_0,
            encoder_layers_1_self_attn_Constant_1_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_q_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0, MatMul_273
        )
        encoder_layers_1_self_attn_q_proj_Add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_q_proj_bias,
            encoder_layers_1_self_attn_q_proj_MatMul_output_0,
        )
        Unsqueeze_176 = opset20.Constant(value_ints=[0])
        encoder_layers_1_self_attn_Unsqueeze_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_Gather_output_0, Unsqueeze_176
        )
        encoder_layers_1_self_attn_Constant_2_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_1_self_attn_Constant_3_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Constant_4_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Concat_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_Unsqueeze_output_0,
            encoder_layers_1_self_attn_Constant_2_output_0,
            encoder_layers_1_self_attn_Constant_3_output_0,
            encoder_layers_1_self_attn_Constant_4_output_0,
            axis=0,
        )
        Unsqueeze_185 = opset20.Constant(value_ints=[0])
        encoder_layers_1_self_attn_Unsqueeze_1_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_Gather_output_0, Unsqueeze_185
        )
        encoder_layers_1_self_attn_Constant_5_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_1_self_attn_Constant_6_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Constant_7_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Concat_1_output_0 = opset20.Constant(
            encoder_layers_1_self_attn_Unsqueeze_1_output_0,
            encoder_layers_1_self_attn_Constant_5_output_0,
            encoder_layers_1_self_attn_Constant_6_output_0,
            encoder_layers_1_self_attn_Constant_7_output_0,
            axis=0,
        )
        Unsqueeze_194 = opset20.Constant(value_ints=[0])
        encoder_layers_1_self_attn_Unsqueeze_2_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_Gather_output_0, Unsqueeze_194
        )
        encoder_layers_1_self_attn_Constant_8_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_1_self_attn_Constant_9_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Constant_10_output_0 = opset20.Constant(value_ints=[4])
        encoder_layers_1_self_attn_Concat_2_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_Unsqueeze_2_output_0,
            encoder_layers_1_self_attn_Constant_8_output_0,
            encoder_layers_1_self_attn_Constant_9_output_0,
            encoder_layers_1_self_attn_Constant_10_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_Reshape_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_q_proj_Add_output_0,
            encoder_layers_1_self_attn_Concat_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_Transpose_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_Reshape_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_1_self_attn_k_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0, MatMul_283
        )
        encoder_layers_1_self_attn_k_proj_Add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_k_proj_bias,
            encoder_layers_1_self_attn_k_proj_MatMul_output_0,
        )
        encoder_layers_1_self_attn_v_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0, MatMul_284
        )
        encoder_layers_1_self_attn_v_proj_Add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_v_proj_bias,
            encoder_layers_1_self_attn_v_proj_MatMul_output_0,
        )
        encoder_layers_1_self_attn_Reshape_1_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_k_proj_Add_output_0,
            encoder_layers_1_self_attn_Concat_1_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_Reshape_2_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_v_proj_Add_output_0,
            encoder_layers_1_self_attn_Concat_2_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_Transpose_1_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_Reshape_2_output_0, perm=[0, 2, 1, 3]
        )
        encoder_layers_1_self_attn_Shape_2_output_0 = opset20.Shape(
            encoder_layers_1_self_attn_Transpose_output_0
        )
        encoder_layers_1_self_attn_Constant_11_output_0 = opset20.Constant(value_ints=[-1])
        encoder_layers_1_self_attn_Constant_12_output_0 = opset20.Constant(
            value_ints=[9223372036854775807]
        )
        encoder_layers_1_self_attn_Slice_output_0 = opset20.Slice(
            encoder_layers_1_self_attn_Shape_2_output_0,
            encoder_layers_1_self_attn_Constant_11_output_0,
            encoder_layers_1_self_attn_Constant_12_output_0,
        )
        encoder_layers_1_self_attn_Cast_output_0 = opset20.Cast(
            encoder_layers_1_self_attn_Slice_output_0, to=1
        )
        encoder_layers_1_self_attn_Sqrt_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_Cast_output_0
        )
        encoder_layers_1_self_attn_Constant_13_output_0 = opset20.Constant(value_floats=[1.0])
        encoder_layers_1_self_attn_Div_output_0 = opset20.Div(
            encoder_layers_1_self_attn_Constant_13_output_0,
            encoder_layers_1_self_attn_Sqrt_output_0,
        )
        encoder_layers_1_self_attn_Cast_1_output_0 = opset20.Cast(
            encoder_layers_1_self_attn_Div_output_0, to=1
        )
        encoder_layers_1_self_attn_Transpose_2_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_Reshape_1_output_0, perm=[0, 2, 3, 1]
        )
        encoder_layers_1_self_attn_Sqrt_1_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_Cast_1_output_0
        )
        encoder_layers_1_self_attn_Mul_output_0 = opset20.Mul(
            encoder_layers_1_self_attn_Transpose_output_0,
            encoder_layers_1_self_attn_Sqrt_1_output_0,
        )
        encoder_layers_1_self_attn_Sqrt_2_output_0 = opset20.Sqrt(
            encoder_layers_1_self_attn_Cast_1_output_0
        )
        encoder_layers_1_self_attn_Mul_1_output_0 = opset20.Mul(
            encoder_layers_1_self_attn_Transpose_2_output_0,
            encoder_layers_1_self_attn_Sqrt_2_output_0,
        )
        encoder_layers_1_self_attn_MatMul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_Mul_output_0, encoder_layers_1_self_attn_Mul_1_output_0
        )
        encoder_layers_1_self_attn_Softmax_output_0 = opset20.Softmax(
            encoder_layers_1_self_attn_MatMul_output_0, axis=-1
        )
        encoder_layers_1_self_attn_MatMul_1_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_Softmax_output_0,
            encoder_layers_1_self_attn_Transpose_1_output_0,
        )
        encoder_layers_1_self_attn_Transpose_3_output_0 = opset20.Transpose(
            encoder_layers_1_self_attn_MatMul_1_output_0, perm=[0, 2, 1, 3]
        )
        Unsqueeze_232 = opset20.Constant(int_values=[0])
        encoder_layers_1_self_attn_Unsqueeze_3_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_Gather_output_0, Unsqueeze_232
        )
        Unsqueeze_234 = opset20.Constant(int_values=[0])
        encoder_layers_1_self_attn_Unsqueeze_4_output_0 = opset20.Unsqueeze(
            encoder_layers_1_self_attn_Gather_1_output_0, Unsqueeze_234
        )
        encoder_layers_1_self_attn_Constant_14_output_0 = opset20.Constant(value_ints=[16])

        encoder_layers_1_self_attn_Concat_3_output_0 = opset20.Concat(
            encoder_layers_1_self_attn_Unsqueeze_3_output_0,
            encoder_layers_1_self_attn_Unsqueeze_4_output_0,
            encoder_layers_1_self_attn_Constant_14_output_0,
            axis=0,
        )
        encoder_layers_1_self_attn_Reshape_3_output_0 = opset20.Reshape(
            encoder_layers_1_self_attn_Transpose_3_output_0,
            encoder_layers_1_self_attn_Concat_3_output_0,
            allowzero=0,
        )
        encoder_layers_1_self_attn_out_proj_MatMul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_Reshape_3_output_0, MatMul_286
        )
        encoder_layers_1_self_attn_out_proj_Add_output_0 = opset20.Add(
            encoder_layers_1_self_attn_out_proj_bias,
            encoder_layers_1_self_attn_out_proj_MatMul_output_0,
        )
        encoder_layers_1_Add_output_0 = opset20.Add(
            encoder_layers_0_final_layer_norm_LayerNormalization_output_0,
            encoder_layers_1_self_attn_out_proj_Add_output_0,
        )
        encoder_layers_1_self_attn_layer_norm_LayerNormalization_output_0 = (
            opset20.LayerNormalization(
                encoder_layers_1_Add_output_0,
                encoder_layers_1_self_attn_layer_norm_weight,
                encoder_layers_1_self_attn_layer_norm_bias,
                axis=-1,
                epsilon=9.999999747378752e-06,
            )
        )
        encoder_layers_1_fc1_MatMul_output_0 = opset20.MatMul(
            encoder_layers_1_self_attn_layer_norm_LayerNormalization_output_0, MatMul_287
        )
        encoder_layers_1_fc1_Add_output_0 = opset20.Add(
            encoder_layers_1_fc1_bias, encoder_layers_1_fc1_MatMul_output_0
        )
        encoder_layers_1_activation_fn_Gelu_output_0 = opset20.Gelu(
            encoder_layers_1_fc1_Add_output_0, approximate="none"
        )
        encoder_layers_1_fc2_MatMul_output_0 = opset20.MatMul(
            encoder_layers_1_activation_fn_Gelu_output_0, MatMul_288
        )
        encoder_layers_1_fc2_Add_output_0 = opset20.Add(
            encoder_layers_1_fc2_bias, encoder_layers_1_fc2_MatMul_output_0
        )
        encoder_layers_1_Add_1_output_0 = opset20.Add(
            encoder_layers_1_self_attn_layer_norm_LayerNormalization_output_0,
            encoder_layers_1_fc2_Add_output_0,
        )
        encoder_output = opset20.LayerNormalization(
            encoder_layers_1_Add_1_output_0,
            encoder_layers_1_final_layer_norm_weight,
            encoder_layers_1_final_layer_norm_bias,
            axis=-1,
            epsilon=9.999999747378752e-06,
        )
        return encoder_output

    return main_graph.to_model_proto()


def make_model_with_random_weights():
    encoder_embed_tokens_weight = np.random.rand(1000, 16).astype(np.float32)
    encoder_embed_positions_weight = np.random.rand(102, 16).astype(np.float32)
    encoder_layers_0_self_attn_k_proj_bias = np.random.rand(16).astype(np.float32)
    encoder_layers_0_self_attn_layer_norm_weight = np.random.rand(16).astype(np.float32)
    encoder_layers_0_fc1_bias = np.zeros((4), dtype=np.float32)

    MatMul_257 = np.random.rand(16, 16).astype(np.float32)
    MatMul_267 = np.random.rand(16, 16).astype(np.float32)
    MatMul_268 = np.random.rand(16, 16).astype(np.float32)
    MatMul_270 = np.random.rand(16, 16).astype(np.float32)
    MatMul_271 = np.random.rand(16, 4).astype(np.float32)
    MatMul_272 = np.random.rand(4, 16).astype(np.float32)
    MatMul_273 = np.random.rand(16, 16).astype(np.float32)
    MatMul_283 = np.random.rand(16, 16).astype(np.float32)
    MatMul_284 = np.random.rand(16, 16).astype(np.float32)
    MatMul_286 = np.random.rand(16, 16).astype(np.float32)
    MatMul_287 = np.random.rand(16, 16).astype(np.float32)
    MatMul_288 = np.random.rand(16, 16).astype(np.float32)

    model = make_model(
        encoder_embed_positions_weight=encoder_embed_positions_weight,
        encoder_embed_tokens_weight=encoder_embed_tokens_weight,
        encoder_layers_0_self_attn_k_proj_bias=encoder_layers_0_self_attn_k_proj_bias,
        encoder_layers_0_self_attn_layer_norm_weight=encoder_layers_0_self_attn_layer_norm_weight,
        encoder_layers_0_fc1_bias=encoder_layers_0_fc1_bias,
        MatMul_257=MatMul_257,
        MatMul_267=MatMul_267,
        MatMul_268=MatMul_268,
        MatMul_270=MatMul_270,
        MatMul_271=MatMul_271,
        MatMul_272=MatMul_272,
        MatMul_273=MatMul_273,
        MatMul_283=MatMul_283,
        MatMul_284=MatMul_284,
        MatMul_286=MatMul_286,
        MatMul_287=MatMul_287,
        MatMul_288=MatMul_288,
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
                "input_ids": np.random.randint(0, 49152, (1, 16)).astype(np.int64),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def bart_encoder_test():
    return _BartEncoderTest()
