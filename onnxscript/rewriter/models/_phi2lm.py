# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Generated from Phi2LM 1 Layer ONNX model produced by the new (Dynamo) exporter
# ruff: noqa: F821

import numpy
import onnx_ir as ir

from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import BOOL, FLOAT, INT64

value_infos = {
    "model_embed_tokens_weight": FLOAT[51200, 2560],
    "model_layers_0_self_attn_q_proj_weight": FLOAT[2560, 2560],
    "model_layers_0_self_attn_q_proj_bias": FLOAT[2560],
    "model_layers_0_self_attn_k_proj_weight": FLOAT[2560, 2560],
    "model_layers_0_self_attn_k_proj_bias": FLOAT[2560],
    "model_layers_0_self_attn_v_proj_weight": FLOAT[2560, 2560],
    "model_layers_0_self_attn_v_proj_bias": FLOAT[2560],
    "model_layers_0_self_attn_dense_weight": FLOAT[2560, 2560],
    "model_layers_0_self_attn_dense_bias": FLOAT[2560],
    "model_layers_0_mlp_fc1_weight": FLOAT[10240, 2560],
    "model_layers_0_mlp_fc1_bias": FLOAT[10240],
    "model_layers_0_mlp_fc2_weight": FLOAT[2560, 10240],
    "model_layers_0_mlp_fc2_bias": FLOAT[2560],
    "model_layers_0_input_layernorm_weight": FLOAT[2560],
    "model_layers_0_input_layernorm_bias": FLOAT[2560],
    "model_final_layernorm_weight": FLOAT[2560],
    "model_final_layernorm_bias": FLOAT[2560],
    "lm_head_weight": FLOAT[51200, 2560],
    "lm_head_bias": FLOAT[51200],
    "expand_2": FLOAT[1, 16, 1],
    "val_1": INT64[1],
    "sym_size_int_44": INT64,
    "val_4": INT64[1],
    "val_5": INT64[1],
    "sym_size_int_50": INT64,
    "embedding": FLOAT["s34", "s16", 2560],
    "add_4": INT64,
    "val_6": FLOAT,
    "val_7": INT64,
    "arange": INT64["s16"],
    "val_8": INT64[1],
    "unsqueeze": INT64[1, "s16"],
    "val_10": FLOAT,
    "val_13": INT64[1],
    "val_14": INT64[1],
    "val_15": INT64[2],
    "full": FLOAT["s16", "s16 + s62"],
    "diagonal": INT64,
    "triu": FLOAT["s16", "s16 + s62"],
    "val_18": INT64,
    "val_19": INT64,
    "arange_1": INT64["s16 + s62"],
    "val_21": INT64[2],
    "view": INT64["s16", 1],
    "gt": BOOL["s16", "s16 + s62"],
    "convert_element_type_default": FLOAT["s16", "s16 + s62"],
    "mul_16": FLOAT["s16", "s16 + s62"],
    "val_22": INT64[1],
    "val_421": INT64[2],
    "unsqueeze_4": FLOAT[1, 1, "s16", "s16 + s62"],
    "val_23": INT64,
    "val_31": INT64,
    "val_49": INT64[1],
    "val_50": INT64[4],
    "val_52": INT64[4],
    "expand_1": FLOAT["s34", 1, "s16", "s16 + s62"],
    "val_61": INT64,
    "val_72": INT64[1],
    "val_74": INT64[1],
    "val_75": INT64[1],
    "val_78": INT64[1],
    "val_79": INT64[1],
    "slice_8": FLOAT["s34", 1, "s16", "s16 + s62"],
    "val_422": INT64[2],
    "unsqueeze_6": INT64["s34", 1, 1, "s16 + s62"],
    "convert_element_type_default_1": FLOAT["s34", 1, 1, "s16 + s62"],
    "add_89": FLOAT["s34", 1, "s16", "s16 + s62"],
    "scalar_tensor_default": FLOAT,
    "eq_64": BOOL["s34", 1, "s16", "s16 + s62"],
    "val_119": INT64[1],
    "val_121": INT64[1],
    "val_122": INT64[1],
    "val_125": INT64[1],
    "val_126": INT64[1],
    "slice_14": FLOAT["s34", 1, "s16", "s16 + s62"],
    "val_127": FLOAT,
    "masked_fill": FLOAT["s34", 1, "s16", "s16 + s62"],
    "val_179": INT64[4],
    "val_180": INT64,
    "val_181": INT64[None],
    "val_186": INT64[None, 1],
    "val_187": FLOAT["s16", 1, "s34", "s16 + s62"],
    "val_188": FLOAT["s16", 1, "s34", "s16 + s62"],
    "val_189": FLOAT["s16", 1, "s34", "s16 + s62"],
    "val_191": INT64[4],
    "val_192": INT64,
    "val_193": INT64[None],
    "val_198": INT64[None, 1],
    "val_199": FLOAT[1, "s34", "s16", "s16 + s62"],
    "val_200": FLOAT[1, "s34", "s16", "s16 + s62"],
    "val_201": FLOAT[1, "s34", "s16", "s16 + s62"],
    "slice_scatter_1": FLOAT["s34", 1, "s16", "s16 + s62"],
    "val_203": INT64[4],
    "val_204": INT64,
    "val_205": INT64[None],
    "val_210": INT64[None, 1],
    "slice_scatter_2": FLOAT["s34", 1, "s16", "s16 + s62"],
    "unsqueeze_9": INT64[1, 1, "s16"],
    "_to_copy": FLOAT[1, 1, "s16"],
    "matmul": FLOAT[1, 16, "s16"],
    "transpose": FLOAT[1, "s16", 16],
    "cat": FLOAT[1, "s16", 32],
    "cos": FLOAT[1, "s16", 32],
    "sin": FLOAT[1, "s16", 32],
    "layer_norm": FLOAT["s34", "s16", 2560],
    "val_246": FLOAT[2560, 2560],
    "val_247": FLOAT["s34", "s16", 2560],
    "linear": FLOAT["s34", "s16", 2560],
    "val_252": INT64[1],
    "val_253": INT64[4],
    "view_1": FLOAT["s34", "s16", 32, 80],
    "transpose_1": FLOAT["s34", 32, "s16", 80],
    "val_255": FLOAT[2560, 2560],
    "val_256": FLOAT["s34", "s16", 2560],
    "linear_1": FLOAT["s34", "s16", 2560],
    "val_261": INT64[4],
    "view_2": FLOAT["s34", "s16", 32, 80],
    "transpose_2": FLOAT["s34", 32, "s16", 80],
    "val_263": FLOAT[2560, 2560],
    "val_264": FLOAT["s34", "s16", 2560],
    "linear_2": FLOAT["s34", "s16", 2560],
    "val_269": INT64[4],
    "view_3": FLOAT["s34", "s16", 32, 80],
    "transpose_3": FLOAT["s34", 32, "s16", 80],
    "val_273": INT64[1],
    "val_277": INT64[1],
    "val_280": INT64[1],
    "val_281": INT64[1],
    "slice_26": FLOAT["s34", 32, "s16", 32],
    "val_284": INT64[1],
    "val_287": INT64[1],
    "val_290": INT64[1],
    "val_291": INT64[1],
    "slice_27": FLOAT["s34", 32, "s16", 48],
    "val_294": INT64[1],
    "val_297": INT64[1],
    "val_300": INT64[1],
    "val_301": INT64[1],
    "slice_28": FLOAT["s34", 32, "s16", 32],
    "val_304": INT64[1],
    "val_307": INT64[1],
    "val_310": INT64[1],
    "val_311": INT64[1],
    "slice_29": FLOAT["s34", 32, "s16", 48],
    "unsqueeze_10": FLOAT[1, 1, "s16", 32],
    "unsqueeze_11": FLOAT[1, 1, "s16", 32],
    "mul_213": FLOAT["s34", 32, "s16", 32],
    "val_314": INT64[1],
    "val_318": INT64[1],
    "val_321": INT64[1],
    "val_322": INT64[1],
    "slice_30": FLOAT["s34", 32, "s16", 16],
    "val_325": INT64[1],
    "val_328": INT64[1],
    "val_331": INT64[1],
    "val_332": INT64[1],
    "slice_31": FLOAT["s34", 32, "s16", 16],
    "neg": FLOAT["s34", 32, "s16", 16],
    "cat_1": FLOAT["s34", 32, "s16", 32],
    "mul_230": FLOAT["s34", 32, "s16", 32],
    "add_290": FLOAT["s34", 32, "s16", 32],
    "mul_238": FLOAT["s34", 32, "s16", 32],
    "val_335": INT64[1],
    "val_338": INT64[1],
    "val_341": INT64[1],
    "val_342": INT64[1],
    "slice_32": FLOAT["s34", 32, "s16", 16],
    "val_345": INT64[1],
    "val_348": INT64[1],
    "val_351": INT64[1],
    "val_352": INT64[1],
    "slice_33": FLOAT["s34", 32, "s16", 16],
    "neg_1": FLOAT["s34", 32, "s16", 16],
    "cat_2": FLOAT["s34", 32, "s16", 32],
    "mul_255": FLOAT["s34", 32, "s16", 32],
    "add_326": FLOAT["s34", 32, "s16", 32],
    "cat_3": FLOAT["s34", 32, "s16", 80],
    "cat_4": FLOAT["s34", 32, "s16", 80],
    "transpose_4": FLOAT["s34", 32, 80, "s16 + s62"],
    "matmul_1": FLOAT["s34", 32, "s16", "s16 + s62"],
    "val_353": FLOAT,
    "mul_287": FLOAT["s34", 32, "s16", "s16 + s62"],
    "val_372": INT64[1],
    "val_374": INT64[1],
    "val_375": INT64[1],
    "val_378": INT64[1],
    "val_379": INT64[1],
    "slice_41": FLOAT["s34", 1, "s16", "s16 + s62"],
    "add_387": FLOAT["s34", 32, "s16", "s16 + s62"],
    "val_380": FLOAT["s34", 32, "s16", "s16 + s62"],
    "matmul_2": FLOAT["s34", 32, "s16", 80],
    "transpose_5": FLOAT["s34", "s16", 32, 80],
    "val_385": INT64[3],
    "view_4": FLOAT["s34", "s16", 2560],
    "val_387": FLOAT[2560, 2560],
    "val_388": FLOAT["s34", "s16", 2560],
    "linear_3": FLOAT["s34", "s16", 2560],
    "val_389": FLOAT[2560, 10240],
    "val_390": FLOAT["s34", "s16", 10240],
    "linear_4": FLOAT["s34", "s16", 10240],
    "val_391": FLOAT,
    "mul_351": FLOAT["s34", "s16", 10240],
    "val_392": FLOAT,
    "pow_1": FLOAT["s34", "s16", 10240],
    "val_393": FLOAT,
    "mul_358": FLOAT["s34", "s16", 10240],
    "add_446": FLOAT["s34", "s16", 10240],
    "val_394": FLOAT,
    "mul_365": FLOAT["s34", "s16", 10240],
    "tanh": FLOAT["s34", "s16", 10240],
    "add_459": FLOAT["s34", "s16", 10240],
    "mul_375": FLOAT["s34", "s16", 10240],
    "val_395": FLOAT[10240, 2560],
    "val_396": FLOAT["s34", "s16", 2560],
    "linear_5": FLOAT["s34", "s16", 2560],
    "add_476": FLOAT["s34", "s16", 2560],
    "add_481": FLOAT["s34", "s16", 2560],
    "layer_norm_1": FLOAT["s34", "s16", 2560],
    "val_419": FLOAT[2560, 51200],
    "val_420": FLOAT["s34", "s16", 51200],
}


def make_model(
    model_embed_tokens_weight,
    model_layers_0_self_attn_q_proj_weight,
    model_layers_0_self_attn_q_proj_bias,
    model_layers_0_self_attn_k_proj_weight,
    model_layers_0_self_attn_k_proj_bias,
    model_layers_0_self_attn_v_proj_weight,
    model_layers_0_self_attn_v_proj_bias,
    model_layers_0_self_attn_dense_weight,
    model_layers_0_self_attn_dense_bias,
    model_layers_0_mlp_fc1_weight,
    model_layers_0_mlp_fc1_bias,
    model_layers_0_mlp_fc2_weight,
    model_layers_0_mlp_fc2_bias,
    model_layers_0_input_layernorm_weight,
    model_layers_0_input_layernorm_bias,
    model_final_layernorm_weight,
    model_final_layernorm_bias,
    lm_head_weight,
    lm_head_bias,
    expand_2,
):
    @script()
    def main_graph(
        input_ids: INT64["s34", "s16"],
        attention_mask: INT64["s34", "s16 + s62"],
        past_key_values_key_cache_0: FLOAT["s34", 32, "s62", 80],
        past_key_values_value_cache_0: FLOAT["s34", 32, "s62", 80],
    ) -> (
        FLOAT["s34", "s16", 51200],
        FLOAT["s34", 32, "s16 + s62", 80],
        FLOAT["s34", 32, "s16 + s62", 80],
    ):
        val_1 = opset18.Shape(input_ids, end=2, start=1)
        sym_size_int_44 = opset18.Squeeze(val_1)
        val_4 = opset18.Shape(past_key_values_value_cache_0, end=1, start=0)
        val_5 = opset18.Shape(past_key_values_value_cache_0, end=3, start=2)
        sym_size_int_50 = opset18.Squeeze(val_5)
        embedding = opset18.Gather(model_embed_tokens_weight, input_ids, axis=0)
        add_4 = opset18.Add(sym_size_int_50, sym_size_int_44)
        arange = opset18.Range(sym_size_int_50, add_4, 1)
        unsqueeze = opset18.Unsqueeze(arange, [0])
        val_14 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_15 = opset18.Concat(val_1, val_14, axis=0)
        full = opset18.Expand(-3.4028235e38, val_15)
        diagonal = opset18.Constant(value_int=1)
        triu = opset18.Trilu(full, diagonal, upper=1)
        arange_1 = opset18.Range(0, add_4, 1)
        view = opset18.Reshape(arange, [-1, 1], allowzero=1)
        gt = opset18.Greater(arange_1, view)
        convert_element_type_default = opset18.Cast(gt, to=1)
        mul_16 = opset18.Mul(triu, convert_element_type_default)
        unsqueeze_4 = opset18.Unsqueeze(mul_16, [0, 1])
        val_50 = opset18.Concat(val_4, [1], [-1], [-1], axis=0)
        val_52 = opset18.Abs(val_50)
        expand_1 = opset18.Expand(unsqueeze_4, val_52)
        val_72 = opset18.Constant(value_ints=[0])
        val_74 = opset18.Constant(value_ints=[-1])
        val_75 = opset18.Reshape(add_4, val_74, allowzero=0)
        val_79 = opset18.Constant(value_ints=[1])
        slice_8 = opset18.Slice(expand_1, val_72, val_75, [3], val_79)
        unsqueeze_6 = opset18.Unsqueeze(attention_mask, [1, 2])
        convert_element_type_default_1 = opset18.Cast(unsqueeze_6, to=1)
        add_89 = opset18.Add(slice_8, convert_element_type_default_1)
        eq_64 = opset18.Equal(add_89, 0.0)
        val_119 = opset18.Constant(value_ints=[0])
        val_121 = opset18.Constant(value_ints=[-1])
        val_122 = opset18.Reshape(add_4, val_121, allowzero=0)
        val_126 = opset18.Constant(value_ints=[1])
        slice_14 = opset18.Slice(expand_1, val_119, val_122, [3], val_126)
        masked_fill = opset18.Where(eq_64, -3.4028235e38, slice_14)
        val_179 = opset18.Shape(expand_1, start=0)
        val_180 = opset18.Gather(val_179, 2, axis=0)
        val_181 = opset18.Range(0, val_180, 1)
        val_186 = opset18.Unsqueeze(val_181, [-1])
        val_187 = opset18.Transpose(masked_fill, perm=[2, 1, 0, 3])
        val_188 = opset18.Transpose(expand_1, perm=[2, 1, 0, 3])
        val_189 = opset18.ScatterND(val_188, val_186, val_187, reduction="none")
        val_191 = opset18.Shape(expand_1, start=0)
        val_192 = opset18.Gather(val_191, 1, axis=0)
        val_193 = opset18.Range(0, val_192, 1)
        val_198 = opset18.Unsqueeze(val_193, [-1])
        val_199 = opset18.Transpose(val_189, perm=[1, 2, 0, 3])
        val_200 = opset18.Transpose(expand_1, perm=[1, 0, 2, 3])
        val_201 = opset18.ScatterND(val_200, val_198, val_199, reduction="none")
        slice_scatter_1 = opset18.Transpose(val_201, perm=[1, 0, 2, 3])
        val_203 = opset18.Shape(expand_1, start=0)
        val_204 = opset18.Gather(val_203, 0, axis=0)
        val_205 = opset18.Range(0, val_204, 1)
        val_210 = opset18.Unsqueeze(val_205, [-1])
        slice_scatter_2 = opset18.ScatterND(
            expand_1, val_210, slice_scatter_1, reduction="none"
        )
        unsqueeze_9 = opset18.Unsqueeze(unsqueeze, [1])
        _to_copy = opset18.Cast(unsqueeze_9, to=1)
        matmul = opset18.MatMul(expand_2, _to_copy)
        transpose = opset18.Transpose(matmul, perm=[0, 2, 1])
        cat = opset18.Concat(transpose, transpose, axis=-1)
        cos = opset18.Cos(cat)
        sin = opset18.Sin(cat)
        layer_norm = opset18.LayerNormalization(
            embedding,
            model_layers_0_input_layernorm_weight,
            model_layers_0_input_layernorm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_246 = opset18.Transpose(model_layers_0_self_attn_q_proj_weight, perm=[1, 0])
        val_247 = opset18.MatMul(layer_norm, val_246)
        linear = opset18.Add(val_247, model_layers_0_self_attn_q_proj_bias)
        val_253 = opset18.Concat(val_4, val_1, [-1], [80], axis=0)
        view_1 = opset18.Reshape(linear, val_253, allowzero=1)
        transpose_1 = opset18.Transpose(view_1, perm=[0, 2, 1, 3])
        val_255 = opset18.Transpose(model_layers_0_self_attn_k_proj_weight, perm=[1, 0])
        val_256 = opset18.MatMul(layer_norm, val_255)
        linear_1 = opset18.Add(val_256, model_layers_0_self_attn_k_proj_bias)
        val_261 = opset18.Concat(val_4, val_1, [-1], [80], axis=0)
        view_2 = opset18.Reshape(linear_1, val_261, allowzero=1)
        transpose_2 = opset18.Transpose(view_2, perm=[0, 2, 1, 3])
        val_263 = opset18.Transpose(model_layers_0_self_attn_v_proj_weight, perm=[1, 0])
        val_264 = opset18.MatMul(layer_norm, val_263)
        linear_2 = opset18.Add(val_264, model_layers_0_self_attn_v_proj_bias)
        val_269 = opset18.Concat(val_4, val_1, [-1], [80], axis=0)
        view_3 = opset18.Reshape(linear_2, val_269, allowzero=1)
        transpose_3 = opset18.Transpose(view_3, perm=[0, 2, 1, 3])
        val_281 = opset18.Constant(value_ints=[1])
        slice_26 = opset18.Slice(transpose_1, [0], [32], [3], val_281)
        val_291 = opset18.Constant(value_ints=[1])
        slice_27 = opset18.Slice(transpose_1, [32], [9223372036854775807], [3], val_291)
        val_301 = opset18.Constant(value_ints=[1])
        slice_28 = opset18.Slice(transpose_2, [0], [32], [3], val_301)
        val_311 = opset18.Constant(value_ints=[1])
        slice_29 = opset18.Slice(transpose_2, [32], [9223372036854775807], [3], val_311)
        unsqueeze_10 = opset18.Unsqueeze(cos, [1])
        unsqueeze_11 = opset18.Unsqueeze(sin, [1])
        mul_213 = opset18.Mul(slice_26, unsqueeze_10)
        val_322 = opset18.Constant(value_ints=[1])
        slice_30 = opset18.Slice(slice_26, [0], [16], [3], val_322)
        val_332 = opset18.Constant(value_ints=[1])
        slice_31 = opset18.Slice(slice_26, [16], [9223372036854775807], [3], val_332)
        neg = opset18.Neg(slice_31)
        cat_1 = opset18.Concat(neg, slice_30, axis=-1)
        mul_230 = opset18.Mul(cat_1, unsqueeze_11)
        add_290 = opset18.Add(mul_213, mul_230)
        mul_238 = opset18.Mul(slice_28, unsqueeze_10)
        val_342 = opset18.Constant(value_ints=[1])
        slice_32 = opset18.Slice(slice_28, [0], [16], [3], val_342)
        val_352 = opset18.Constant(value_ints=[1])
        slice_33 = opset18.Slice(slice_28, [16], [9223372036854775807], [3], val_352)
        neg_1 = opset18.Neg(slice_33)
        cat_2 = opset18.Concat(neg_1, slice_32, axis=-1)
        mul_255 = opset18.Mul(cat_2, unsqueeze_11)
        add_326 = opset18.Add(mul_238, mul_255)
        cat_3 = opset18.Concat(add_290, slice_27, axis=-1)
        cat_4 = opset18.Concat(add_326, slice_29, axis=-1)
        cat_5 = opset18.Concat(past_key_values_key_cache_0, cat_4, axis=-2)
        cat_6 = opset18.Concat(past_key_values_value_cache_0, transpose_3, axis=-2)
        transpose_4 = opset18.Transpose(cat_5, perm=[0, 1, 3, 2])
        matmul_1 = opset18.MatMul(cat_3, transpose_4)
        mul_287 = opset18.Mul(matmul_1, 0.1118034)
        val_372 = opset18.Constant(value_ints=[0])
        val_374 = opset18.Constant(value_ints=[-1])
        val_375 = opset18.Reshape(add_4, val_374, allowzero=0)
        val_379 = opset18.Constant(value_ints=[1])
        slice_41 = opset18.Slice(slice_scatter_2, val_372, val_375, [3], val_379)
        add_387 = opset18.Add(mul_287, slice_41)
        val_380 = opset18.Softmax(add_387, axis=-1)
        matmul_2 = opset18.MatMul(val_380, cat_6)
        transpose_5 = opset18.Transpose(matmul_2, perm=[0, 2, 1, 3])
        val_385 = opset18.Concat(val_4, val_1, [-1], axis=0)
        view_4 = opset18.Reshape(transpose_5, val_385, allowzero=1)
        val_387 = opset18.Transpose(model_layers_0_self_attn_dense_weight, perm=[1, 0])
        val_388 = opset18.MatMul(view_4, val_387)
        linear_3 = opset18.Add(val_388, model_layers_0_self_attn_dense_bias)
        val_389 = opset18.Transpose(model_layers_0_mlp_fc1_weight, perm=[1, 0])
        val_390 = opset18.MatMul(layer_norm, val_389)
        linear_4 = opset18.Add(val_390, model_layers_0_mlp_fc1_bias)
        mul_351 = opset18.Mul(linear_4, 0.5)
        pow_1 = opset18.Pow(linear_4, 3.0)
        mul_358 = opset18.Mul(pow_1, 0.044715)
        add_446 = opset18.Add(linear_4, mul_358)
        mul_365 = opset18.Mul(add_446, 0.7978846)
        tanh = opset18.Tanh(mul_365)
        add_459 = opset18.Add(tanh, 1.0)
        mul_375 = opset18.Mul(mul_351, add_459)
        val_395 = opset18.Transpose(model_layers_0_mlp_fc2_weight, perm=[1, 0])
        val_396 = opset18.MatMul(mul_375, val_395)
        linear_5 = opset18.Add(val_396, model_layers_0_mlp_fc2_bias)
        add_476 = opset18.Add(linear_3, linear_5)
        add_481 = opset18.Add(add_476, embedding)
        layer_norm_1 = opset18.LayerNormalization(
            add_481,
            model_final_layernorm_weight,
            model_final_layernorm_bias,
            stash_type=1,
            epsilon=9.999999747378752e-06,
            axis=-1,
        )
        val_419 = opset18.Transpose(lm_head_weight, perm=[1, 0])
        val_420 = opset18.MatMul(layer_norm_1, val_419)
        linear_6 = opset18.Add(val_420, lm_head_bias)
        return linear_6, cat_5, cat_6

    model = main_graph.to_model_proto(value_infos=value_infos)
    return model


def make_model_with_random_weights():
    model_embed_tokens_weight = numpy.random.rand(51200, 2560).astype(numpy.float32)
    model_layers_0_self_attn_q_proj_weight = numpy.random.rand(2560, 2560).astype(
        numpy.float32
    )
    model_layers_0_self_attn_q_proj_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_self_attn_k_proj_weight = numpy.random.rand(2560, 2560).astype(
        numpy.float32
    )
    model_layers_0_self_attn_k_proj_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_self_attn_v_proj_weight = numpy.random.rand(2560, 2560).astype(
        numpy.float32
    )
    model_layers_0_self_attn_v_proj_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_self_attn_dense_weight = numpy.random.rand(2560, 2560).astype(numpy.float32)
    model_layers_0_self_attn_dense_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_mlp_fc1_weight = numpy.random.rand(10240, 2560).astype(numpy.float32)
    model_layers_0_mlp_fc1_bias = numpy.random.rand(10240).astype(numpy.float32)
    model_layers_0_mlp_fc2_weight = numpy.random.rand(2560, 10240).astype(numpy.float32)
    model_layers_0_mlp_fc2_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_input_layernorm_weight = numpy.random.rand(2560).astype(numpy.float32)
    model_layers_0_input_layernorm_bias = numpy.random.rand(2560).astype(numpy.float32)
    model_final_layernorm_weight = numpy.random.rand(2560).astype(numpy.float32)
    model_final_layernorm_bias = numpy.random.rand(2560).astype(numpy.float32)
    lm_head_weight = numpy.random.rand(51200, 2560).astype(numpy.float32)
    lm_head_bias = numpy.random.rand(51200).astype(numpy.float32)
    expand_2 = numpy.random.rand(1, 16, 1).astype(numpy.float32)
    model = make_model(
        model_embed_tokens_weight,
        model_layers_0_self_attn_q_proj_weight,
        model_layers_0_self_attn_q_proj_bias,
        model_layers_0_self_attn_k_proj_weight,
        model_layers_0_self_attn_k_proj_bias,
        model_layers_0_self_attn_v_proj_weight,
        model_layers_0_self_attn_v_proj_bias,
        model_layers_0_self_attn_dense_weight,
        model_layers_0_self_attn_dense_bias,
        model_layers_0_mlp_fc1_weight,
        model_layers_0_mlp_fc1_bias,
        model_layers_0_mlp_fc2_weight,
        model_layers_0_mlp_fc2_bias,
        model_layers_0_input_layernorm_weight,
        model_layers_0_input_layernorm_bias,
        model_final_layernorm_weight,
        model_final_layernorm_bias,
        lm_head_weight,
        lm_head_bias,
        expand_2,
    )
    return model


class _Phi2LMTest:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model


def phi2lm_test():
    return _Phi2LMTest()
