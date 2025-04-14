# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A one-layer SmolLM model test case, with inputs: input_ids, position_ids, and pask key/values.
This is an onnxscript version of the model.
"""

import numpy

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT, INT64


def make_model(
    model_layers_0_input_layernorm_weight,
    model_layers_0_post_attention_layernorm_weight,
    model_norm_weight,
    lm_head_weight,
    model_layers_0_self_attn_q_proj_weight,
    model_layers_0_self_attn_k_proj_weight,
    model_layers_0_self_attn_v_proj_weight,
    model_layers_0_self_attn_o_proj_weight,
    model_layers_0_mlp_gate_proj_weight,
    model_layers_0_mlp_up_proj_weight,
    model_layers_0_mlp_down_proj_weight,
    model_rotary_emb_inv_freq,
):
    @script()
    def main_graph(
        input_ids: INT64[1, 30],
        position_ids: INT64[1, 30],
        past_key_values_0_0: FLOAT[1, 32, 16, 64],
        past_key_values_0_1: FLOAT[1, 32, 16, 64],
    ) -> (FLOAT[1, 30, 49152], FLOAT[1, 32, 46, 64], FLOAT[1, 32, 46, 64]):
        embedding = opset18.Gather(lm_head_weight, input_ids, axis=0)
        val_2 = opset18.CastLike(1.0, 46)
        arange = opset18.Range(16, 46, val_2)
        val_5 = opset18.Cast(-3.4028235e38, to=1)
        val_7 = opset18.Cast([30, 47], to=7)
        full = opset18.Expand(val_5, val_7)
        diagonal__1 = opset18.Constant(value_int=1)
        triu = opset18.Trilu(full, diagonal__1, upper=1)
        val_10 = opset18.CastLike(0.0, 47)
        val_11 = opset18.CastLike(1.0, 47)
        arange_1 = opset18.Range(val_10, 47, val_11)
        val_13 = opset18.Cast([-1, 1], to=7)
        view = opset18.Reshape(arange, val_13, allowzero=0)
        gt = arange_1 > view
        convert_element_type_default = opset18.Cast(gt, to=1)
        mul = triu * convert_element_type_default
        dim__2 = opset18.Constant(value_int=0)
        dim_0__2 = opset18.Cast(dim__2, to=7)
        unsqueeze = opset18.Unsqueeze(model_rotary_emb_inv_freq, dim_0__2)
        val_15 = opset18.Cast(0, to=7)
        val_16 = opset18.Constant(value_ints=[-1])
        val_17 = opset18.Reshape(val_15, val_16, allowzero=0)
        val_19 = opset18.Cast(9223372036854775807, to=7)
        val_20 = opset18.Constant(value_ints=[-1])
        val_21 = opset18.Reshape(val_19, val_20, allowzero=0)
        val_23 = opset18.Cast(1, to=7)
        val_24 = opset18.Constant(value_ints=[-1])
        val_25 = opset18.Reshape(val_23, val_24, allowzero=0)
        val_26 = opset18.Constant(value_ints=[1])
        slice_1 = opset18.Slice(unsqueeze, val_17, val_21, val_25, val_26)
        dim__3 = opset18.Constant(value_int=2)
        dim_0__3 = opset18.Cast(dim__3, to=7)
        unsqueeze_1 = opset18.Unsqueeze(slice_1, dim_0__3)
        _to_copy = opset18.Cast(unsqueeze_1, to=1)
        size_0__4 = opset18.Cast([1, -1, 1], to=7)
        size_1__4 = opset18.Abs(size_0__4)
        expand = opset18.Expand(_to_copy, size_1__4)
        val_28 = opset18.Cast(0, to=7)
        val_29 = opset18.Constant(value_ints=[-1])
        val_30 = opset18.Reshape(val_28, val_29, allowzero=0)
        val_31 = opset18.Cast(9223372036854775807, to=7)
        val_32 = opset18.Constant(value_ints=[-1])
        val_33 = opset18.Reshape(val_31, val_32, allowzero=0)
        val_34 = opset18.Cast(0, to=7)
        val_35 = opset18.Constant(value_ints=[-1])
        val_36 = opset18.Reshape(val_34, val_35, allowzero=0)
        val_37 = opset18.Constant(value_ints=[1])
        slice_2 = opset18.Slice(position_ids, val_30, val_33, val_36, val_37)
        dim__5 = opset18.Constant(value_int=1)
        dim_0__5 = opset18.Cast(dim__5, to=7)
        unsqueeze_2 = opset18.Unsqueeze(slice_2, dim_0__5)
        val_38 = opset18.Cast(0, to=7)
        val_39 = opset18.Constant(value_ints=[-1])
        val_40 = opset18.Reshape(val_38, val_39, allowzero=0)
        val_41 = opset18.Cast(9223372036854775807, to=7)
        val_42 = opset18.Constant(value_ints=[-1])
        val_43 = opset18.Reshape(val_41, val_42, allowzero=0)
        val_45 = opset18.Cast(2, to=7)
        val_46 = opset18.Constant(value_ints=[-1])
        val_47 = opset18.Reshape(val_45, val_46, allowzero=0)
        val_48 = opset18.Constant(value_ints=[1])
        slice_3 = opset18.Slice(unsqueeze_2, val_40, val_43, val_47, val_48)
        _to_copy_1 = opset18.Cast(slice_3, to=1)
        _to_copy_2 = opset18.Cast(expand, to=1)
        _to_copy_3 = opset18.Cast(_to_copy_1, to=1)
        size_0__6 = opset18.Cast([1, 32, 1], to=7)
        size_1__6 = opset18.Abs(size_0__6)
        expand_1 = opset18.Expand(_to_copy_2, size_1__6)
        val_50 = opset18.Cast([1, 32, 1], to=7)
        view_1 = opset18.Reshape(expand_1, val_50, allowzero=0)
        size_0__7 = opset18.Cast([1, 1, 30], to=7)
        size_1__7 = opset18.Abs(size_0__7)
        expand_2 = opset18.Expand(_to_copy_3, size_1__7)
        val_52 = opset18.Cast([1, 1, 30], to=7)
        view_2 = opset18.Reshape(expand_2, val_52, allowzero=0)
        bmm = view_1 @ view_2
        val_54 = opset18.Cast([1, 32, 30], to=7)
        view_3 = opset18.Reshape(bmm, val_54, allowzero=0)
        transpose = opset18.Transpose(view_3, perm=[0, 2, 1])
        cat = opset18.Concat(transpose, transpose, axis=-1)
        cos = opset18.Cos(cat)
        sin = opset18.Sin(cat)
        mul_1 = cos * 1.0
        mul_2 = sin * 1.0
        _to_copy_4 = opset18.Cast(mul_1, to=1)
        _to_copy_5 = opset18.Cast(mul_2, to=1)
        _to_copy_6 = opset18.Cast(embedding, to=1)
        scalar_tensor_default = opset18.Cast(2, to=1)
        pow_1 = _to_copy_6**scalar_tensor_default
        val_55 = opset18.Constant(value_ints=[-1])
        val_57 = opset18.Reshape([-1], val_55, allowzero=0)
        mean = opset18.ReduceMean(pow_1, val_57, keepdims=1, noop_with_empty_axes=0)
        add = mean + 1e-05
        val_59 = opset18.Sqrt(add)
        rsqrt = opset18.Reciprocal(val_59)
        mul_3 = _to_copy_6 * rsqrt
        _to_copy_7 = opset18.Cast(mul_3, to=1)
        mul_4 = model_layers_0_input_layernorm_weight * _to_copy_7
        t = opset18.Transpose(model_layers_0_self_attn_q_proj_weight, perm=[1, 0])
        val_61 = opset18.Cast([30, 2048], to=7)
        view_4 = opset18.Reshape(mul_4, val_61, allowzero=0)
        mm = view_4 @ t
        val_63 = opset18.Cast([1, 30, 2048], to=7)
        view_5 = opset18.Reshape(mm, val_63, allowzero=0)
        t_1 = opset18.Transpose(model_layers_0_self_attn_k_proj_weight, perm=[1, 0])
        val_64 = opset18.Cast([30, 2048], to=7)
        view_6 = opset18.Reshape(mul_4, val_64, allowzero=0)
        mm_1 = view_6 @ t_1
        val_65 = opset18.Cast([1, 30, 2048], to=7)
        view_7 = opset18.Reshape(mm_1, val_65, allowzero=0)
        t_2 = opset18.Transpose(model_layers_0_self_attn_v_proj_weight, perm=[1, 0])
        val_66 = opset18.Cast([30, 2048], to=7)
        view_8 = opset18.Reshape(mul_4, val_66, allowzero=0)
        mm_2 = view_8 @ t_2
        val_67 = opset18.Cast([1, 30, 2048], to=7)
        view_9 = opset18.Reshape(mm_2, val_67, allowzero=0)
        val_69 = opset18.Cast([1, 30, 32, 64], to=7)
        view_10 = opset18.Reshape(view_5, val_69, allowzero=0)
        transpose_1 = opset18.Transpose(view_10, perm=[0, 2, 1, 3])
        val_70 = opset18.Cast([1, 30, 32, 64], to=7)
        view_11 = opset18.Reshape(view_7, val_70, allowzero=0)
        transpose_2 = opset18.Transpose(view_11, perm=[0, 2, 1, 3])
        val_71 = opset18.Cast([1, 30, 32, 64], to=7)
        view_12 = opset18.Reshape(view_9, val_71, allowzero=0)
        transpose_3 = opset18.Transpose(view_12, perm=[0, 2, 1, 3])
        dim__8 = opset18.Constant(value_int=1)
        dim_0__8 = opset18.Cast(dim__8, to=7)
        unsqueeze_3 = opset18.Unsqueeze(_to_copy_4, dim_0__8)
        dim__9 = opset18.Constant(value_int=1)
        dim_0__9 = opset18.Cast(dim__9, to=7)
        unsqueeze_4 = opset18.Unsqueeze(_to_copy_5, dim_0__9)
        mul_5 = transpose_1 * unsqueeze_3
        val_72 = opset18.Cast(0, to=7)
        val_73 = opset18.Constant(value_ints=[-1])
        val_74 = opset18.Reshape(val_72, val_73, allowzero=0)
        val_76 = opset18.Cast(32, to=7)
        val_77 = opset18.Constant(value_ints=[-1])
        val_78 = opset18.Reshape(val_76, val_77, allowzero=0)
        val_80 = opset18.Cast(3, to=7)
        val_81 = opset18.Constant(value_ints=[-1])
        val_82 = opset18.Reshape(val_80, val_81, allowzero=0)
        val_83 = opset18.Constant(value_ints=[1])
        slice_4 = opset18.Slice(transpose_1, val_74, val_78, val_82, val_83)
        val_84 = opset18.Cast(32, to=7)
        val_85 = opset18.Constant(value_ints=[-1])
        val_86 = opset18.Reshape(val_84, val_85, allowzero=0)
        val_87 = opset18.Cast(9223372036854775807, to=7)
        val_88 = opset18.Constant(value_ints=[-1])
        val_89 = opset18.Reshape(val_87, val_88, allowzero=0)
        val_90 = opset18.Cast(3, to=7)
        val_91 = opset18.Constant(value_ints=[-1])
        val_92 = opset18.Reshape(val_90, val_91, allowzero=0)
        val_93 = opset18.Constant(value_ints=[1])
        slice_5 = opset18.Slice(transpose_1, val_86, val_89, val_92, val_93)
        neg = opset18.Neg(slice_5)
        cat_1 = opset18.Concat(neg, slice_4, axis=-1)
        mul_6 = cat_1 * unsqueeze_4
        add_1 = mul_5 + mul_6
        mul_7 = transpose_2 * unsqueeze_3
        val_94 = opset18.Cast(0, to=7)
        val_95 = opset18.Constant(value_ints=[-1])
        val_96 = opset18.Reshape(val_94, val_95, allowzero=0)
        val_97 = opset18.Cast(32, to=7)
        val_98 = opset18.Constant(value_ints=[-1])
        val_99 = opset18.Reshape(val_97, val_98, allowzero=0)
        val_100 = opset18.Cast(3, to=7)
        val_101 = opset18.Constant(value_ints=[-1])
        val_102 = opset18.Reshape(val_100, val_101, allowzero=0)
        val_103 = opset18.Constant(value_ints=[1])
        slice_6 = opset18.Slice(transpose_2, val_96, val_99, val_102, val_103)
        val_104 = opset18.Cast(32, to=7)
        val_105 = opset18.Constant(value_ints=[-1])
        val_106 = opset18.Reshape(val_104, val_105, allowzero=0)
        val_107 = opset18.Cast(9223372036854775807, to=7)
        val_108 = opset18.Constant(value_ints=[-1])
        val_109 = opset18.Reshape(val_107, val_108, allowzero=0)
        val_110 = opset18.Cast(3, to=7)
        val_111 = opset18.Constant(value_ints=[-1])
        val_112 = opset18.Reshape(val_110, val_111, allowzero=0)
        val_113 = opset18.Constant(value_ints=[1])
        slice_7 = opset18.Slice(transpose_2, val_106, val_109, val_112, val_113)
        neg_1 = opset18.Neg(slice_7)
        cat_2 = opset18.Concat(neg_1, slice_6, axis=-1)
        mul_8 = cat_2 * unsqueeze_4
        add_2 = mul_7 + mul_8
        cat_3 = opset18.Concat(past_key_values_0_0, add_2, axis=-2)
        cat_4 = opset18.Concat(past_key_values_0_1, transpose_3, axis=-2)
        dim__10 = opset18.Constant(value_int=0)
        dim_0__10 = opset18.Cast(dim__10, to=7)
        unsqueeze_5 = opset18.Unsqueeze(mul, dim_0__10)
        dim__11 = opset18.Constant(value_int=1)
        dim_0__11 = opset18.Cast(dim__11, to=7)
        unsqueeze_6 = opset18.Unsqueeze(unsqueeze_5, dim_0__11)
        val_114 = opset18.Cast(0, to=7)
        val_115 = opset18.Constant(value_ints=[-1])
        val_116 = opset18.Reshape(val_114, val_115, allowzero=0)
        val_117 = opset18.Cast(9223372036854775807, to=7)
        val_118 = opset18.Constant(value_ints=[-1])
        val_119 = opset18.Reshape(val_117, val_118, allowzero=0)
        val_120 = opset18.Cast(2, to=7)
        val_121 = opset18.Constant(value_ints=[-1])
        val_122 = opset18.Reshape(val_120, val_121, allowzero=0)
        val_123 = opset18.Constant(value_ints=[1])
        slice_8 = opset18.Slice(unsqueeze_6, val_116, val_119, val_122, val_123)
        val_124 = opset18.Cast(0, to=7)
        val_125 = opset18.Constant(value_ints=[-1])
        val_126 = opset18.Reshape(val_124, val_125, allowzero=0)
        val_127 = opset18.Cast(9223372036854775807, to=7)
        val_128 = opset18.Constant(value_ints=[-1])
        val_129 = opset18.Reshape(val_127, val_128, allowzero=0)
        val_130 = opset18.Cast(3, to=7)
        val_131 = opset18.Constant(value_ints=[-1])
        val_132 = opset18.Reshape(val_130, val_131, allowzero=0)
        val_133 = opset18.Constant(value_ints=[1])
        slice_9 = opset18.Slice(slice_8, val_126, val_129, val_132, val_133)
        size_0__12 = opset18.Cast([1, 1, -1, -1], to=7)
        size_1__12 = opset18.Abs(size_0__12)
        expand_3 = opset18.Expand(slice_9, size_1__12)
        val_135 = opset18.Cast(0, to=7)
        val_136 = opset18.Constant(value_ints=[-1])
        val_137 = opset18.Reshape(val_135, val_136, allowzero=0)
        val_138 = opset18.Cast(9223372036854775807, to=7)
        val_139 = opset18.Constant(value_ints=[-1])
        val_140 = opset18.Reshape(val_138, val_139, allowzero=0)
        val_141 = opset18.Cast(0, to=7)
        val_142 = opset18.Constant(value_ints=[-1])
        val_143 = opset18.Reshape(val_141, val_142, allowzero=0)
        val_144 = opset18.Constant(value_ints=[1])
        slice_10 = opset18.Slice(expand_3, val_137, val_140, val_143, val_144)
        val_145 = opset18.Cast(0, to=7)
        val_146 = opset18.Constant(value_ints=[-1])
        val_147 = opset18.Reshape(val_145, val_146, allowzero=0)
        val_148 = opset18.Cast(9223372036854775807, to=7)
        val_149 = opset18.Constant(value_ints=[-1])
        val_150 = opset18.Reshape(val_148, val_149, allowzero=0)
        val_151 = opset18.Cast(1, to=7)
        val_152 = opset18.Constant(value_ints=[-1])
        val_153 = opset18.Reshape(val_151, val_152, allowzero=0)
        val_154 = opset18.Constant(value_ints=[1])
        slice_11 = opset18.Slice(slice_10, val_147, val_150, val_153, val_154)
        val_155 = opset18.Cast(0, to=7)
        val_156 = opset18.Constant(value_ints=[-1])
        val_157 = opset18.Reshape(val_155, val_156, allowzero=0)
        val_158 = opset18.Cast(9223372036854775807, to=7)
        val_159 = opset18.Constant(value_ints=[-1])
        val_160 = opset18.Reshape(val_158, val_159, allowzero=0)
        val_161 = opset18.Cast(2, to=7)
        val_162 = opset18.Constant(value_ints=[-1])
        val_163 = opset18.Reshape(val_161, val_162, allowzero=0)
        val_164 = opset18.Constant(value_ints=[1])
        slice_12 = opset18.Slice(slice_11, val_157, val_160, val_163, val_164)
        val_165 = opset18.Cast(0, to=7)
        val_166 = opset18.Constant(value_ints=[-1])
        val_167 = opset18.Reshape(val_165, val_166, allowzero=0)
        val_168 = opset18.Cast(46, to=7)
        val_169 = opset18.Constant(value_ints=[-1])
        val_170 = opset18.Reshape(val_168, val_169, allowzero=0)
        val_171 = opset18.Cast(3, to=7)
        val_172 = opset18.Constant(value_ints=[-1])
        val_173 = opset18.Reshape(val_171, val_172, allowzero=0)
        val_174 = opset18.Constant(value_ints=[1])
        slice_13 = opset18.Slice(slice_12, val_167, val_170, val_173, val_174)
        val_175 = opset18.Shape(add_1, start=0)
        val_176 = opset18.Constant(value_ints=[-1])
        val_177 = opset18.Gather(val_175, val_176, axis=0)
        val_178 = opset18.CastLike(val_177, add_1)
        val_179 = opset18.Constant(value_float=1.0)
        val_180 = opset18.CastLike(val_179, add_1)
        val_181 = opset18.Sqrt(val_178)
        val_182 = val_180 / val_181
        val_183 = opset18.CastLike(val_182, add_1)
        val_184 = opset18.Shape(cat_3, start=0)
        val_185 = opset18.Constant(value_ints=[9223372036854775807])
        val_186 = opset18.Slice(val_184, [-1], val_185)
        val_188 = opset18.Slice(val_184, [-2], [-1])
        val_189 = opset18.Constant(value_ints=[-9223372036854775808])
        val_190 = opset18.Slice(val_184, val_189, [-2])
        val_191 = opset18.Constant(value_ints=[-1])
        val_192 = opset18.Concat(val_191, val_188, val_186, axis=0)
        val_193 = opset18.Reshape(cat_3, val_192, allowzero=0)
        val_194 = opset18.Transpose(val_193, perm=[0, 2, 1])
        val_195 = opset18.Concat(val_190, val_186, val_188, axis=0)
        val_196 = opset18.Reshape(val_194, val_195, allowzero=0)
        val_197 = opset18.Sqrt(val_183)
        val_198 = add_1 * val_197
        val_199 = opset18.Sqrt(val_183)
        val_200 = val_196 * val_199
        val_201 = val_198 @ val_200
        val_202 = val_201 + slice_13
        val_203 = opset18.Softmax(val_202, axis=-1)
        val_204, _unused = opset18.Dropout(val_203, 0.0)
        getitem = val_204 @ cat_4
        val_206 = opset18.Shape(add_1, start=0)
        val_209 = opset18.Slice(val_206, [0], [1])
        val_211 = opset18.Slice(val_206, [1], [2])
        val_212 = opset18.Slice(val_206, [-2], [-1])
        val_213 = opset18.Cast(val_211, to=1)
        val_215 = val_213 / 32.0
        val_216 = opset18.Ceil(val_215)
        val_217 = val_216 * 32.0
        val_218 = opset18.Cast(val_217, to=7)
        val_219 = opset18.Concat(val_209, val_212, val_218, axis=0)
        _scaled_dot_product_flash_attention_for_cpu__1 = opset18.Expand(0.0, val_219)
        transpose_4 = opset18.Transpose(getitem, perm=[0, 2, 1, 3])
        val_221 = opset18.Cast([1, 30, -1], to=7)
        view_13 = opset18.Reshape(transpose_4, val_221, allowzero=0)
        t_3 = opset18.Transpose(model_layers_0_self_attn_o_proj_weight, perm=[1, 0])
        val_222 = opset18.Cast([30, 2048], to=7)
        view_14 = opset18.Reshape(view_13, val_222, allowzero=0)
        mm_3 = view_14 @ t_3
        val_223 = opset18.Cast([1, 30, 2048], to=7)
        view_15 = opset18.Reshape(mm_3, val_223, allowzero=0)
        add_3 = embedding + view_15
        _to_copy_8 = opset18.Cast(add_3, to=1)
        scalar_tensor_default_1 = opset18.Cast(2, to=1)
        pow_2 = _to_copy_8**scalar_tensor_default_1
        val_224 = opset18.Constant(value_ints=[-1])
        val_225 = opset18.Reshape([-1], val_224, allowzero=0)
        mean_1 = opset18.ReduceMean(pow_2, val_225, keepdims=1, noop_with_empty_axes=0)
        add_4 = mean_1 + 1e-05
        val_226 = opset18.Sqrt(add_4)
        rsqrt_1 = opset18.Reciprocal(val_226)
        mul_9 = _to_copy_8 * rsqrt_1
        _to_copy_9 = opset18.Cast(mul_9, to=1)
        mul_10 = model_layers_0_post_attention_layernorm_weight * _to_copy_9
        t_4 = opset18.Transpose(model_layers_0_mlp_gate_proj_weight, perm=[1, 0])
        val_227 = opset18.Cast([30, 2048], to=7)
        view_16 = opset18.Reshape(mul_10, val_227, allowzero=0)
        mm_4 = view_16 @ t_4
        val_229 = opset18.Cast([1, 30, 8192], to=7)
        view_17 = opset18.Reshape(mm_4, val_229, allowzero=0)
        val_230 = opset18.Sigmoid(view_17)
        silu = view_17 * val_230
        t_5 = opset18.Transpose(model_layers_0_mlp_up_proj_weight, perm=[1, 0])
        val_231 = opset18.Cast([30, 2048], to=7)
        view_18 = opset18.Reshape(mul_10, val_231, allowzero=0)
        mm_5 = view_18 @ t_5
        val_232 = opset18.Cast([1, 30, 8192], to=7)
        view_19 = opset18.Reshape(mm_5, val_232, allowzero=0)
        mul_11 = silu * view_19
        t_6 = opset18.Transpose(model_layers_0_mlp_down_proj_weight, perm=[1, 0])
        val_234 = opset18.Cast([30, 8192], to=7)
        view_20 = opset18.Reshape(mul_11, val_234, allowzero=0)
        mm_6 = view_20 @ t_6
        val_235 = opset18.Cast([1, 30, 2048], to=7)
        view_21 = opset18.Reshape(mm_6, val_235, allowzero=0)
        add_5 = add_3 + view_21
        _to_copy_10 = opset18.Cast(add_5, to=1)
        scalar_tensor_default_2 = opset18.Cast(2, to=1)
        pow_3 = _to_copy_10**scalar_tensor_default_2
        val_236 = opset18.Constant(value_ints=[-1])
        val_237 = opset18.Reshape([-1], val_236, allowzero=0)
        mean_2 = opset18.ReduceMean(pow_3, val_237, keepdims=1, noop_with_empty_axes=0)
        add_6 = mean_2 + 1e-05
        val_238 = opset18.Sqrt(add_6)
        rsqrt_2 = opset18.Reciprocal(val_238)
        mul_12 = _to_copy_10 * rsqrt_2
        _to_copy_11 = opset18.Cast(mul_12, to=1)
        mul_13 = model_norm_weight * _to_copy_11
        t_7 = opset18.Transpose(lm_head_weight, perm=[1, 0])
        val_239 = opset18.Cast([30, 2048], to=7)
        view_22 = opset18.Reshape(mul_13, val_239, allowzero=0)
        mm_7 = view_22 @ t_7
        val_241 = opset18.Cast([1, 30, 49152], to=7)
        view_23 = opset18.Reshape(mm_7, val_241, allowzero=0)
        _to_copy_12 = opset18.Cast(view_23, to=1)
        return _to_copy_12, cat_3, cat_4

    model = main_graph.to_model_proto()
    return model


def make_model_with_random_weights():
    model_layers_0_input_layernorm_weight = numpy.random.rand(2048).astype(numpy.float32)
    model_layers_0_post_attention_layernorm_weight = numpy.random.rand(2048).astype(
        numpy.float32
    )
    model_norm_weight = numpy.random.rand(2048).astype(numpy.float32)
    lm_head_weight = numpy.random.rand(49152, 2048).astype(numpy.float32)
    model_layers_0_self_attn_q_proj_weight = numpy.random.rand(2048, 2048).astype(
        numpy.float32
    )
    model_layers_0_self_attn_k_proj_weight = numpy.random.rand(2048, 2048).astype(
        numpy.float32
    )
    model_layers_0_self_attn_v_proj_weight = numpy.random.rand(2048, 2048).astype(
        numpy.float32
    )
    model_layers_0_self_attn_o_proj_weight = numpy.random.rand(2048, 2048).astype(
        numpy.float32
    )
    model_layers_0_mlp_gate_proj_weight = numpy.random.rand(8192, 2048).astype(numpy.float32)
    model_layers_0_mlp_up_proj_weight = numpy.random.rand(8192, 2048).astype(numpy.float32)
    model_layers_0_mlp_down_proj_weight = numpy.random.rand(2048, 8192).astype(numpy.float32)
    model_rotary_emb_inv_freq = numpy.random.rand(32).astype(numpy.float32)
    model = make_model(
        model_layers_0_input_layernorm_weight,
        model_layers_0_post_attention_layernorm_weight,
        model_norm_weight,
        lm_head_weight,
        model_layers_0_self_attn_q_proj_weight,
        model_layers_0_self_attn_k_proj_weight,
        model_layers_0_self_attn_v_proj_weight,
        model_layers_0_self_attn_o_proj_weight,
        model_layers_0_mlp_gate_proj_weight,
        model_layers_0_mlp_up_proj_weight,
        model_layers_0_mlp_down_proj_weight,
        model_rotary_emb_inv_freq,
    )
    return model


class _SmollmTest2:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "input_ids": numpy.random.randint(0, 49152, (1, 30)).astype(numpy.int64),
                "position_ids": numpy.arange(30).reshape(1, 30).astype(numpy.int64),
                "past_key_values_0_0": numpy.random.rand(1, 32, 16, 64).astype(numpy.float32),
                "past_key_values_0_1": numpy.random.rand(1, 32, 16, 64).astype(numpy.float32),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def smollm_test_2():
    return _SmollmTest2()
