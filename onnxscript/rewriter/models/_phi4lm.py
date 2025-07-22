# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Generated from Phi4LM 2 Layer ONNX model produced by the new (Dynamo) exporter
# ruff: noqa: F821

import numpy
import onnx_ir as ir

from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import BOOL, FLOAT, INT64

value_infos = {
    "model_embed_tokens_weight": FLOAT[100352, 5120],
    "model_layers_0_self_attn_o_proj_weight": FLOAT[5120, 5120],
    "model_layers_0_self_attn_qkv_proj_weight": FLOAT[7680, 5120],
    "model_layers_0_mlp_gate_up_proj_weight": FLOAT[35840, 5120],
    "model_layers_0_mlp_down_proj_weight": FLOAT[5120, 17920],
    "model_layers_0_input_layernorm_weight": FLOAT[5120],
    "model_layers_0_post_attention_layernorm_weight": FLOAT[5120],
    "model_layers_1_self_attn_o_proj_weight": FLOAT[5120, 5120],
    "model_layers_1_self_attn_qkv_proj_weight": FLOAT[7680, 5120],
    "model_layers_1_mlp_gate_up_proj_weight": FLOAT[35840, 5120],
    "model_layers_1_mlp_down_proj_weight": FLOAT[5120, 17920],
    "model_layers_1_input_layernorm_weight": FLOAT[5120],
    "model_layers_1_post_attention_layernorm_weight": FLOAT[5120],
    "model_norm_weight": FLOAT[5120],
    "lm_head_weight": FLOAT[100352, 5120],
    "expand_2": FLOAT[1, 64, 1],
    "val_1": INT64[1],
    "sym_size_int_61": INT64,
    "val_5": INT64[1],
    "sym_size_int_67": INT64,
    "val_6": INT64[1],
    "embedding": FLOAT["s34", "s16", 5120],
    "add_4": INT64,
    "val_11": INT64,
    "arange": INT64["s16"],
    "val_12": INT64[1],
    "unsqueeze": INT64[1, "s16"],
    "val_14": FLOAT,
    "val_17": INT64[1],
    "val_18": INT64[1],
    "val_19": INT64[2],
    "full": FLOAT["s16", "s16 + s17"],
    "val_22": INT64,
    "val_23": INT64,
    "arange_1": INT64["s16 + s17"],
    "val_25": INT64[2],
    "view": INT64["s16", 1],
    "gt": BOOL["s16", "s16 + s17"],
    "convert_element_type_default": FLOAT["s16", "s16 + s17"],
    "mul_14": FLOAT["s16", "s16 + s17"],
    "val_26": INT64[1],
    "val_805": INT64[2],
    "unsqueeze_4": FLOAT[1, 1, "s16", "s16 + s17"],
    "val_27": INT64,
    "val_35": INT64,
    "val_53": INT64[1],
    "val_54": INT64[4],
    "val_56": INT64[4],
    "expand_1": FLOAT["s34", 1, "s16", "s16 + s17"],
    "val_65": INT64,
    "val_76": INT64[1],
    "val_78": INT64[1],
    "val_79": INT64[1],
    "val_82": INT64[1],
    "val_83": INT64[1],
    "slice_8": FLOAT["s34", 1, "s16", "s16 + s17"],
    "val_94": INT64[1],
    "val_806": INT64[2],
    "unsqueeze_6": INT64["s34", 1, 1, "s16 + s17"],
    "convert_element_type_default_1": FLOAT["s34", 1, 1, "s16 + s17"],
    "add_86": FLOAT["s34", 1, "s16", "s16 + s17"],
    "scalar_tensor_default": FLOAT,
    "eq_65": BOOL["s34", 1, "s16", "s16 + s17"],
    "val_123": INT64[1],
    "val_125": INT64[1],
    "val_126": INT64[1],
    "val_129": INT64[1],
    "val_130": INT64[1],
    "slice_14": FLOAT["s34", 1, "s16", "s16 + s17"],
    "val_131": FLOAT,
    "masked_fill": FLOAT["s34", 1, "s16", "s16 + s17"],
    "val_183": INT64[4],
    "val_184": INT64,
    "val_185": INT64[None],
    "val_190": INT64[None, 1],
    "val_191": FLOAT["s16", 1, "s34", "s16 + s17"],
    "val_192": FLOAT["s16", 1, "s34", "s16 + s17"],
    "val_193": FLOAT["s16", 1, "s34", "s16 + s17"],
    "val_195": INT64[4],
    "val_196": INT64,
    "val_197": INT64[None],
    "val_202": INT64[None, 1],
    "val_203": FLOAT[1, "s34", "s16", "s16 + s17"],
    "val_204": FLOAT[1, "s34", "s16", "s16 + s17"],
    "val_205": FLOAT[1, "s34", "s16", "s16 + s17"],
    "slice_scatter_1": FLOAT["s34", 1, "s16", "s16 + s17"],
    "val_207": INT64[4],
    "val_208": INT64,
    "val_209": INT64[None],
    "val_214": INT64[None, 1],
    "slice_scatter_2": FLOAT["s34", 1, "s16", "s16 + s17"],
    "unsqueeze_9": INT64[1, 1, "s16"],
    "_to_copy": FLOAT[1, 1, "s16"],
    "matmul": FLOAT[1, 64, "s16"],
    "transpose": FLOAT[1, "s16", 64],
    "cat": FLOAT[1, "s16", 128],
    "cos": FLOAT[1, "s16", 128],
    "sin": FLOAT[1, "s16", 128],
    "val_248": FLOAT,
    "pow_1": FLOAT["s34", "s16", 5120],
    "val_250": INT64[1],
    "mean": FLOAT["s34", "s16", 1],
    "val_251": FLOAT,
    "add_189": FLOAT["s34", "s16", 1],
    "val_252": FLOAT["s34", "s16", 1],
    "rsqrt": FLOAT["s34", "s16", 1],
    "mul_167": FLOAT["s34", "s16", 5120],
    "mul_171": FLOAT["s34", "s16", 5120],
    "val_253": FLOAT[5120, 7680],
    "linear": FLOAT["s34", "s16", 7680],
    "val_256": INT64[1],
    "val_260": INT64[1],
    "val_263": INT64[1],
    "val_264": INT64[1],
    "slice_26": FLOAT["s34", "s16", 5120],
    "val_267": INT64[1],
    "val_271": INT64[1],
    "val_274": INT64[1],
    "val_275": INT64[1],
    "slice_27": FLOAT["s34", "s16", 1280],
    "val_278": INT64[1],
    "val_281": INT64[1],
    "val_284": INT64[1],
    "val_285": INT64[1],
    "slice_28": FLOAT["s34", "s16", 1280],
    "val_290": INT64[1],
    "val_291": INT64[4],
    "view_1": FLOAT["s34", "s16", 40, 128],
    "transpose_1": FLOAT["s34", 40, "s16", 128],
    "val_297": INT64[4],
    "view_2": FLOAT["s34", "s16", 10, 128],
    "transpose_2": FLOAT["s34", 10, "s16", 128],
    "val_303": INT64[4],
    "view_3": FLOAT["s34", "s16", 10, 128],
    "transpose_3": FLOAT["s34", 10, "s16", 128],
    "unsqueeze_10": FLOAT[1, 1, "s16", 128],
    "unsqueeze_11": FLOAT[1, 1, "s16", 128],
    "mul_223": FLOAT["s34", 40, "s16", 128],
    "val_328": INT64[1],
    "val_332": INT64[1],
    "val_335": INT64[1],
    "val_336": INT64[1],
    "slice_31": FLOAT["s34", 40, "s16", 64],
    "val_339": INT64[1],
    "val_342": INT64[1],
    "val_345": INT64[1],
    "val_346": INT64[1],
    "slice_32": FLOAT["s34", 40, "s16", 64],
    "neg": FLOAT["s34", 40, "s16", 64],
    "cat_1": FLOAT["s34", 40, "s16", 128],
    "mul_240": FLOAT["s34", 40, "s16", 128],
    "add_304": FLOAT["s34", 40, "s16", 128],
    "mul_252": FLOAT["s34", 10, "s16", 128],
    "val_349": INT64[1],
    "val_352": INT64[1],
    "val_355": INT64[1],
    "val_356": INT64[1],
    "slice_33": FLOAT["s34", 10, "s16", 64],
    "val_359": INT64[1],
    "val_362": INT64[1],
    "val_365": INT64[1],
    "val_366": INT64[1],
    "slice_34": FLOAT["s34", 10, "s16", 64],
    "neg_1": FLOAT["s34", 10, "s16", 64],
    "cat_3": FLOAT["s34", 10, "s16", 128],
    "mul_269": FLOAT["s34", 10, "s16", 128],
    "add_345": FLOAT["s34", 10, "s16", 128],
    "unsqueeze_12": FLOAT["s34", 10, 1, "s16 + s17", 128],
    "val_410": INT64[1],
    "val_411": INT64[1],
    "val_412": INT64[1],
    "val_413": INT64[1],
    "val_414": INT64[5],
    "val_416": INT64[5],
    "expand_3": FLOAT["s34", 10, 4, "s16 + s17", 128],
    "val_419": INT64[1],
    "val_420": INT64[1],
    "val_421": INT64[1],
    "val_422": INT64[4],
    "_unsafe_view": FLOAT["s34", 40, "s16 + s17", 128],
    "unsqueeze_13": FLOAT["s34", 10, 1, "s16 + s17", 128],
    "val_466": INT64[1],
    "val_467": INT64[1],
    "val_468": INT64[5],
    "val_470": INT64[5],
    "expand_4": FLOAT["s34", 10, 4, "s16 + s17", 128],
    "val_473": INT64[1],
    "val_474": INT64[1],
    "val_475": INT64[4],
    "_unsafe_view_1": FLOAT["s34", 40, "s16 + s17", 128],
    "transpose_4": FLOAT["s34", 40, 128, "s16 + s17"],
    "matmul_1": FLOAT["s34", 40, "s16", "s16 + s17"],
    "val_477": FLOAT,
    "mul_433": FLOAT["s34", 40, "s16", "s16 + s17"],
    "val_496": INT64[1],
    "val_498": INT64[1],
    "val_499": INT64[1],
    "val_502": INT64[1],
    "val_503": INT64[1],
    "slice_50": FLOAT["s34", 1, "s16", "s16 + s17"],
    "add_491": FLOAT["s34", 40, "s16", "s16 + s17"],
    "val_504": FLOAT["s34", 40, "s16", "s16 + s17"],
    "matmul_2": FLOAT["s34", 40, "s16", 128],
    "transpose_5": FLOAT["s34", "s16", 40, 128],
    "val_509": INT64[3],
    "view_4": FLOAT["s34", "s16", 5120],
    "val_511": FLOAT[5120, 5120],
    "linear_1": FLOAT["s34", "s16", 5120],
    "add_534": FLOAT["s34", "s16", 5120],
    "val_512": FLOAT,
    "pow_2": FLOAT["s34", "s16", 5120],
    "val_514": INT64[1],
    "mean_1": FLOAT["s34", "s16", 1],
    "add_547": FLOAT["s34", "s16", 1],
    "val_515": FLOAT["s34", "s16", 1],
    "rsqrt_1": FLOAT["s34", "s16", 1],
    "mul_506": FLOAT["s34", "s16", 5120],
    "mul_510": FLOAT["s34", "s16", 5120],
    "val_516": FLOAT[5120, 35840],
    "linear_2": FLOAT["s34", "s16", 35840],
    "split_split_0": FLOAT["s34", "s16", 17920],
    "split_split_1": FLOAT["s34", "s16", 17920],
    "val_518": FLOAT["s34", "s16", 17920],
    "silu": FLOAT["s34", "s16", 17920],
    "mul_526": FLOAT["s34", "s16", 17920],
    "val_519": FLOAT[17920, 5120],
    "linear_3": FLOAT["s34", "s16", 5120],
    "add_592": FLOAT["s34", "s16", 5120],
    "val_520": FLOAT,
    "pow_3": FLOAT["s34", "s16", 5120],
    "val_522": INT64[1],
    "mean_2": FLOAT["s34", "s16", 1],
    "add_605": FLOAT["s34", "s16", 1],
    "val_523": FLOAT["s34", "s16", 1],
    "rsqrt_2": FLOAT["s34", "s16", 1],
    "mul_548": FLOAT["s34", "s16", 5120],
    "mul_552": FLOAT["s34", "s16", 5120],
    "val_524": FLOAT[5120, 7680],
    "linear_4": FLOAT["s34", "s16", 7680],
    "val_527": INT64[1],
    "val_530": INT64[1],
    "val_533": INT64[1],
    "val_534": INT64[1],
    "slice_51": FLOAT["s34", "s16", 5120],
    "val_537": INT64[1],
    "val_540": INT64[1],
    "val_543": INT64[1],
    "val_544": INT64[1],
    "slice_52": FLOAT["s34", "s16", 1280],
    "val_547": INT64[1],
    "val_550": INT64[1],
    "val_553": INT64[1],
    "val_554": INT64[1],
    "slice_53": FLOAT["s34", "s16", 1280],
    "val_559": INT64[4],
    "view_5": FLOAT["s34", "s16", 40, 128],
    "transpose_6": FLOAT["s34", 40, "s16", 128],
    "val_565": INT64[4],
    "view_6": FLOAT["s34", "s16", 10, 128],
    "transpose_7": FLOAT["s34", 10, "s16", 128],
    "val_571": INT64[4],
    "view_7": FLOAT["s34", "s16", 10, 128],
    "transpose_8": FLOAT["s34", 10, "s16", 128],
    "unsqueeze_14": FLOAT[1, 1, "s16", 128],
    "unsqueeze_15": FLOAT[1, 1, "s16", 128],
    "mul_604": FLOAT["s34", 40, "s16", 128],
    "val_595": INT64[1],
    "val_598": INT64[1],
    "val_601": INT64[1],
    "val_602": INT64[1],
    "slice_56": FLOAT["s34", 40, "s16", 64],
    "val_605": INT64[1],
    "val_608": INT64[1],
    "val_611": INT64[1],
    "val_612": INT64[1],
    "slice_57": FLOAT["s34", 40, "s16", 64],
    "neg_2": FLOAT["s34", 40, "s16", 64],
    "cat_7": FLOAT["s34", 40, "s16", 128],
    "mul_621": FLOAT["s34", 40, "s16", 128],
    "add_720": FLOAT["s34", 40, "s16", 128],
    "mul_633": FLOAT["s34", 10, "s16", 128],
    "val_615": INT64[1],
    "val_618": INT64[1],
    "val_621": INT64[1],
    "val_622": INT64[1],
    "slice_58": FLOAT["s34", 10, "s16", 64],
    "val_625": INT64[1],
    "val_628": INT64[1],
    "val_631": INT64[1],
    "val_632": INT64[1],
    "slice_59": FLOAT["s34", 10, "s16", 64],
    "neg_3": FLOAT["s34", 10, "s16", 64],
    "cat_9": FLOAT["s34", 10, "s16", 128],
    "mul_650": FLOAT["s34", 10, "s16", 128],
    "add_761": FLOAT["s34", 10, "s16", 128],
    "unsqueeze_16": FLOAT["s34", 10, 1, "s16 + s17", 128],
    "val_675": INT64[1],
    "val_676": INT64[1],
    "val_677": INT64[5],
    "val_679": INT64[5],
    "expand_5": FLOAT["s34", 10, 4, "s16 + s17", 128],
    "val_682": INT64[1],
    "val_683": INT64[1],
    "val_684": INT64[4],
    "_unsafe_view_2": FLOAT["s34", 40, "s16 + s17", 128],
    "unsqueeze_17": FLOAT["s34", 10, 1, "s16 + s17", 128],
    "val_728": INT64[1],
    "val_729": INT64[1],
    "val_730": INT64[5],
    "val_732": INT64[5],
    "expand_6": FLOAT["s34", 10, 4, "s16 + s17", 128],
    "val_735": INT64[1],
    "val_736": INT64[1],
    "val_737": INT64[4],
    "_unsafe_view_3": FLOAT["s34", 40, "s16 + s17", 128],
    "transpose_9": FLOAT["s34", 40, 128, "s16 + s17"],
    "matmul_3": FLOAT["s34", 40, "s16", "s16 + s17"],
    "mul_814": FLOAT["s34", 40, "s16", "s16 + s17"],
    "val_757": INT64[1],
    "val_759": INT64[1],
    "val_760": INT64[1],
    "val_763": INT64[1],
    "val_764": INT64[1],
    "slice_75": FLOAT["s34", 1, "s16", "s16 + s17"],
    "add_907": FLOAT["s34", 40, "s16", "s16 + s17"],
    "val_765": FLOAT["s34", 40, "s16", "s16 + s17"],
    "matmul_4": FLOAT["s34", 40, "s16", 128],
    "transpose_10": FLOAT["s34", "s16", 40, 128],
    "val_770": INT64[3],
    "view_8": FLOAT["s34", "s16", 5120],
    "val_772": FLOAT[5120, 5120],
    "linear_5": FLOAT["s34", "s16", 5120],
    "add_950": FLOAT["s34", "s16", 5120],
    "val_773": FLOAT,
    "pow_4": FLOAT["s34", "s16", 5120],
    "val_775": INT64[1],
    "mean_3": FLOAT["s34", "s16", 1],
    "add_963": FLOAT["s34", "s16", 1],
    "val_776": FLOAT["s34", "s16", 1],
    "rsqrt_3": FLOAT["s34", "s16", 1],
    "mul_887": FLOAT["s34", "s16", 5120],
    "mul_891": FLOAT["s34", "s16", 5120],
    "val_777": FLOAT[5120, 35840],
    "linear_6": FLOAT["s34", "s16", 35840],
    "split_1_split_0": FLOAT["s34", "s16", 17920],
    "split_1_split_1": FLOAT["s34", "s16", 17920],
    "val_778": FLOAT["s34", "s16", 17920],
    "silu_1": FLOAT["s34", "s16", 17920],
    "mul_907": FLOAT["s34", "s16", 17920],
    "val_779": FLOAT[17920, 5120],
    "linear_7": FLOAT["s34", "s16", 5120],
    "add_1008": FLOAT["s34", "s16", 5120],
    "val_780": FLOAT,
    "pow_5": FLOAT["s34", "s16", 5120],
    "val_782": INT64[1],
    "mean_4": FLOAT["s34", "s16", 1],
    "add_1021": FLOAT["s34", "s16", 1],
    "val_783": FLOAT["s34", "s16", 1],
    "rsqrt_4": FLOAT["s34", "s16", 1],
    "mul_929": FLOAT["s34", "s16", 5120],
    "mul_933": FLOAT["s34", "s16", 5120],
    "val_804": FLOAT[5120, 100352],
}


def make_model(
    model_embed_tokens_weight,
    model_layers_0_self_attn_o_proj_weight,
    model_layers_0_self_attn_qkv_proj_weight,
    model_layers_0_mlp_gate_up_proj_weight,
    model_layers_0_mlp_down_proj_weight,
    model_layers_0_input_layernorm_weight,
    model_layers_0_post_attention_layernorm_weight,
    model_layers_1_self_attn_o_proj_weight,
    model_layers_1_self_attn_qkv_proj_weight,
    model_layers_1_mlp_gate_up_proj_weight,
    model_layers_1_mlp_down_proj_weight,
    model_layers_1_input_layernorm_weight,
    model_layers_1_post_attention_layernorm_weight,
    model_norm_weight,
    lm_head_weight,
    expand_2,
):
    @script()
    def main_graph(
        input_ids: INT64["s34", "s16"],
        attention_mask: INT64["s34", "s16 + s17"],
        past_key_values_key_cache_0: FLOAT["s34", 10, "s17", 128],
        past_key_values_key_cache_1: FLOAT["s34", 10, "s17", 128],
        past_key_values_value_cache_0: FLOAT["s34", 10, "s17", 128],
        past_key_values_value_cache_1: FLOAT["s34", 10, "s17", 128],
    ) -> (
        FLOAT["s34", "s16", 100352],
        FLOAT["s34", 10, "s16 + s17", 128],
        FLOAT["s34", 10, "s16 + s17", 128],
        FLOAT["s34", 10, "s16 + s17", 128],
        FLOAT["s34", 10, "s16 + s17", 128],
    ):
        val_1 = opset18.Shape(input_ids, end=2, start=1)
        sym_size_int_61 = opset18.Squeeze(val_1)
        val_5 = opset18.Shape(past_key_values_key_cache_1, end=3, start=2)
        sym_size_int_67 = opset18.Squeeze(val_5)
        val_6 = opset18.Shape(past_key_values_value_cache_0, end=1, start=0)
        embedding = opset18.Gather(model_embed_tokens_weight, input_ids, axis=0)
        add_4 = opset18.Add(sym_size_int_67, sym_size_int_61)
        arange = opset18.Range(sym_size_int_67, add_4, 1)
        unsqueeze = opset18.Unsqueeze(arange, [0])
        val_18 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_19 = opset18.Concat(val_1, val_18, axis=0)
        full = opset18.Expand(-3.4028235e38, val_19)
        arange_1 = opset18.Range(0, add_4, 1)
        view = opset18.Reshape(arange, [-1, 1], allowzero=1)
        gt = opset18.Greater(arange_1, view)
        convert_element_type_default = opset18.Cast(gt, to=1)
        mul_14 = opset18.Mul(full, convert_element_type_default)
        unsqueeze_4 = opset18.Unsqueeze(mul_14, [0, 1])
        val_54 = opset18.Concat(val_6, [1], [-1], [-1], axis=0)
        val_56 = opset18.Abs(val_54)
        expand_1 = opset18.Expand(unsqueeze_4, val_56)
        val_76 = opset18.Constant(value_ints=[0])
        val_78 = opset18.Constant(value_ints=[-1])
        val_79 = opset18.Reshape(add_4, val_78, allowzero=0)
        val_83 = opset18.Constant(value_ints=[1])
        slice_8 = opset18.Slice(expand_1, val_76, val_79, [3], val_83)
        unsqueeze_6 = opset18.Unsqueeze(attention_mask, [1, 2])
        convert_element_type_default_1 = opset18.Cast(unsqueeze_6, to=1)
        add_86 = opset18.Add(slice_8, convert_element_type_default_1)
        eq_65 = opset18.Equal(add_86, 0.0)
        val_123 = opset18.Constant(value_ints=[0])
        val_125 = opset18.Constant(value_ints=[-1])
        val_126 = opset18.Reshape(add_4, val_125, allowzero=0)
        val_130 = opset18.Constant(value_ints=[1])
        slice_14 = opset18.Slice(expand_1, val_123, val_126, [3], val_130)
        masked_fill = opset18.Where(eq_65, -3.4028235e38, slice_14)
        val_183 = opset18.Shape(expand_1, start=0)
        val_184 = opset18.Gather(val_183, 2, axis=0)
        val_185 = opset18.Range(0, val_184, 1)
        val_190 = opset18.Unsqueeze(val_185, [-1])
        val_191 = opset18.Transpose(masked_fill, perm=[2, 1, 0, 3])
        val_192 = opset18.Transpose(expand_1, perm=[2, 1, 0, 3])
        val_193 = opset18.ScatterND(val_192, val_190, val_191, reduction="none")
        val_195 = opset18.Shape(expand_1, start=0)
        val_196 = opset18.Gather(val_195, 1, axis=0)
        val_197 = opset18.Range(0, val_196, 1)
        val_202 = opset18.Unsqueeze(val_197, [-1])
        val_203 = opset18.Transpose(val_193, perm=[1, 2, 0, 3])
        val_204 = opset18.Transpose(expand_1, perm=[1, 0, 2, 3])
        val_205 = opset18.ScatterND(val_204, val_202, val_203, reduction="none")
        slice_scatter_1 = opset18.Transpose(val_205, perm=[1, 0, 2, 3])
        val_207 = opset18.Shape(expand_1, start=0)
        val_208 = opset18.Gather(val_207, 0, axis=0)
        val_209 = opset18.Range(0, val_208, 1)
        val_214 = opset18.Unsqueeze(val_209, [-1])
        slice_scatter_2 = opset18.ScatterND(
            expand_1, val_214, slice_scatter_1, reduction="none"
        )
        unsqueeze_9 = opset18.Unsqueeze(unsqueeze, [1])
        _to_copy = opset18.Cast(unsqueeze_9, to=1)
        matmul = opset18.MatMul(expand_2, _to_copy)
        transpose = opset18.Transpose(matmul, perm=[0, 2, 1])
        cat = opset18.Concat(transpose, transpose, axis=-1)
        cos = opset18.Cos(cat)
        sin = opset18.Sin(cat)
        pow_1 = opset18.Pow(embedding, 2.0)
        mean = opset18.ReduceMean(pow_1, [-1], noop_with_empty_axes=0, keepdims=1)
        add_189 = opset18.Add(mean, 1e-05)
        val_252 = opset18.Sqrt(add_189)
        rsqrt = opset18.Reciprocal(val_252)
        mul_167 = opset18.Mul(embedding, rsqrt)
        mul_171 = opset18.Mul(model_layers_0_input_layernorm_weight, mul_167)
        val_253 = opset18.Transpose(model_layers_0_self_attn_qkv_proj_weight, perm=[1, 0])
        linear = opset18.MatMul(mul_171, val_253)
        val_264 = opset18.Constant(value_ints=[1])
        slice_26 = opset18.Slice(linear, [0], [5120], [2], val_264)
        val_275 = opset18.Constant(value_ints=[1])
        slice_27 = opset18.Slice(linear, [5120], [6400], [2], val_275)
        val_285 = opset18.Constant(value_ints=[1])
        slice_28 = opset18.Slice(linear, [6400], [9223372036854775807], [2], val_285)
        val_291 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_1 = opset18.Reshape(slice_26, val_291, allowzero=1)
        transpose_1 = opset18.Transpose(view_1, perm=[0, 2, 1, 3])
        val_297 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_2 = opset18.Reshape(slice_27, val_297, allowzero=1)
        transpose_2 = opset18.Transpose(view_2, perm=[0, 2, 1, 3])
        val_303 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_3 = opset18.Reshape(slice_28, val_303, allowzero=1)
        transpose_3 = opset18.Transpose(view_3, perm=[0, 2, 1, 3])
        unsqueeze_10 = opset18.Unsqueeze(cos, [1])
        unsqueeze_11 = opset18.Unsqueeze(sin, [1])
        mul_223 = opset18.Mul(transpose_1, unsqueeze_10)
        val_336 = opset18.Constant(value_ints=[1])
        slice_31 = opset18.Slice(transpose_1, [0], [64], [3], val_336)
        val_346 = opset18.Constant(value_ints=[1])
        slice_32 = opset18.Slice(transpose_1, [64], [9223372036854775807], [3], val_346)
        neg = opset18.Neg(slice_32)
        cat_1 = opset18.Concat(neg, slice_31, axis=-1)
        mul_240 = opset18.Mul(cat_1, unsqueeze_11)
        add_304 = opset18.Add(mul_223, mul_240)
        mul_252 = opset18.Mul(transpose_2, unsqueeze_10)
        val_356 = opset18.Constant(value_ints=[1])
        slice_33 = opset18.Slice(transpose_2, [0], [64], [3], val_356)
        val_366 = opset18.Constant(value_ints=[1])
        slice_34 = opset18.Slice(transpose_2, [64], [9223372036854775807], [3], val_366)
        neg_1 = opset18.Neg(slice_34)
        cat_3 = opset18.Concat(neg_1, slice_33, axis=-1)
        mul_269 = opset18.Mul(cat_3, unsqueeze_11)
        add_345 = opset18.Add(mul_252, mul_269)
        cat_5 = opset18.Concat(past_key_values_key_cache_0, add_345, axis=-2)
        cat_6 = opset18.Concat(past_key_values_value_cache_0, transpose_3, axis=-2)
        unsqueeze_12 = opset18.Unsqueeze(cat_5, [2])
        val_413 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_414 = opset18.Concat(val_6, [10], [4], val_413, [128], axis=0)
        val_416 = opset18.Abs(val_414)
        expand_3 = opset18.Expand(unsqueeze_12, val_416)
        val_421 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_422 = opset18.Concat(val_6, [40], val_421, [128], axis=0)
        _unsafe_view = opset18.Reshape(expand_3, val_422, allowzero=1)
        unsqueeze_13 = opset18.Unsqueeze(cat_6, [2])
        val_467 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_468 = opset18.Concat(val_6, [10], [4], val_467, [128], axis=0)
        val_470 = opset18.Abs(val_468)
        expand_4 = opset18.Expand(unsqueeze_13, val_470)
        val_474 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_475 = opset18.Concat(val_6, [40], val_474, [128], axis=0)
        _unsafe_view_1 = opset18.Reshape(expand_4, val_475, allowzero=1)
        transpose_4 = opset18.Transpose(_unsafe_view, perm=[0, 1, 3, 2])
        matmul_1 = opset18.MatMul(add_304, transpose_4)
        mul_433 = opset18.Mul(matmul_1, 0.088388346)
        val_496 = opset18.Constant(value_ints=[0])
        val_498 = opset18.Constant(value_ints=[-1])
        val_499 = opset18.Reshape(add_4, val_498, allowzero=0)
        val_503 = opset18.Constant(value_ints=[1])
        slice_50 = opset18.Slice(slice_scatter_2, val_496, val_499, [3], val_503)
        add_491 = opset18.Add(mul_433, slice_50)
        val_504 = opset18.Softmax(add_491, axis=-1)
        matmul_2 = opset18.MatMul(val_504, _unsafe_view_1)
        transpose_5 = opset18.Transpose(matmul_2, perm=[0, 2, 1, 3])
        val_509 = opset18.Concat(val_6, val_1, [-1], axis=0)
        view_4 = opset18.Reshape(transpose_5, val_509, allowzero=1)
        val_511 = opset18.Transpose(model_layers_0_self_attn_o_proj_weight, perm=[1, 0])
        linear_1 = opset18.MatMul(view_4, val_511)
        add_534 = opset18.Add(embedding, linear_1)
        pow_2 = opset18.Pow(add_534, 2.0)
        mean_1 = opset18.ReduceMean(pow_2, [-1], noop_with_empty_axes=0, keepdims=1)
        add_547 = opset18.Add(mean_1, 1e-05)
        val_515 = opset18.Sqrt(add_547)
        rsqrt_1 = opset18.Reciprocal(val_515)
        mul_506 = opset18.Mul(add_534, rsqrt_1)
        mul_510 = opset18.Mul(model_layers_0_post_attention_layernorm_weight, mul_506)
        val_516 = opset18.Transpose(model_layers_0_mlp_gate_up_proj_weight, perm=[1, 0])
        linear_2 = opset18.MatMul(mul_510, val_516)
        split_split_0, split_split_1 = opset18.Split(linear_2, axis=2, num_outputs=2)
        val_518 = opset18.Sigmoid(split_split_0)
        silu = opset18.Mul(split_split_0, val_518)
        mul_526 = opset18.Mul(split_split_1, silu)
        val_519 = opset18.Transpose(model_layers_0_mlp_down_proj_weight, perm=[1, 0])
        linear_3 = opset18.MatMul(mul_526, val_519)
        add_592 = opset18.Add(add_534, linear_3)
        pow_3 = opset18.Pow(add_592, 2.0)
        mean_2 = opset18.ReduceMean(pow_3, [-1], noop_with_empty_axes=0, keepdims=1)
        add_605 = opset18.Add(mean_2, 1e-05)
        val_523 = opset18.Sqrt(add_605)
        rsqrt_2 = opset18.Reciprocal(val_523)
        mul_548 = opset18.Mul(add_592, rsqrt_2)
        mul_552 = opset18.Mul(model_layers_1_input_layernorm_weight, mul_548)
        val_524 = opset18.Transpose(model_layers_1_self_attn_qkv_proj_weight, perm=[1, 0])
        linear_4 = opset18.MatMul(mul_552, val_524)
        val_534 = opset18.Constant(value_ints=[1])
        slice_51 = opset18.Slice(linear_4, [0], [5120], [2], val_534)
        val_544 = opset18.Constant(value_ints=[1])
        slice_52 = opset18.Slice(linear_4, [5120], [6400], [2], val_544)
        val_554 = opset18.Constant(value_ints=[1])
        slice_53 = opset18.Slice(linear_4, [6400], [9223372036854775807], [2], val_554)
        val_559 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_5 = opset18.Reshape(slice_51, val_559, allowzero=1)
        transpose_6 = opset18.Transpose(view_5, perm=[0, 2, 1, 3])
        val_565 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_6 = opset18.Reshape(slice_52, val_565, allowzero=1)
        transpose_7 = opset18.Transpose(view_6, perm=[0, 2, 1, 3])
        val_571 = opset18.Concat(val_6, val_1, [-1], [128], axis=0)
        view_7 = opset18.Reshape(slice_53, val_571, allowzero=1)
        transpose_8 = opset18.Transpose(view_7, perm=[0, 2, 1, 3])
        unsqueeze_14 = opset18.Unsqueeze(cos, [1])
        unsqueeze_15 = opset18.Unsqueeze(sin, [1])
        mul_604 = opset18.Mul(transpose_6, unsqueeze_14)
        val_602 = opset18.Constant(value_ints=[1])
        slice_56 = opset18.Slice(transpose_6, [0], [64], [3], val_602)
        val_612 = opset18.Constant(value_ints=[1])
        slice_57 = opset18.Slice(transpose_6, [64], [9223372036854775807], [3], val_612)
        neg_2 = opset18.Neg(slice_57)
        cat_7 = opset18.Concat(neg_2, slice_56, axis=-1)
        mul_621 = opset18.Mul(cat_7, unsqueeze_15)
        add_720 = opset18.Add(mul_604, mul_621)
        mul_633 = opset18.Mul(transpose_7, unsqueeze_14)
        val_622 = opset18.Constant(value_ints=[1])
        slice_58 = opset18.Slice(transpose_7, [0], [64], [3], val_622)
        val_632 = opset18.Constant(value_ints=[1])
        slice_59 = opset18.Slice(transpose_7, [64], [9223372036854775807], [3], val_632)
        neg_3 = opset18.Neg(slice_59)
        cat_9 = opset18.Concat(neg_3, slice_58, axis=-1)
        mul_650 = opset18.Mul(cat_9, unsqueeze_15)
        add_761 = opset18.Add(mul_633, mul_650)
        cat_11 = opset18.Concat(past_key_values_key_cache_1, add_761, axis=-2)
        cat_12 = opset18.Concat(past_key_values_value_cache_1, transpose_8, axis=-2)
        unsqueeze_16 = opset18.Unsqueeze(cat_11, [2])
        val_676 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_677 = opset18.Concat(val_6, [10], [4], val_676, [128], axis=0)
        val_679 = opset18.Abs(val_677)
        expand_5 = opset18.Expand(unsqueeze_16, val_679)
        val_683 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_684 = opset18.Concat(val_6, [40], val_683, [128], axis=0)
        _unsafe_view_2 = opset18.Reshape(expand_5, val_684, allowzero=1)
        unsqueeze_17 = opset18.Unsqueeze(cat_12, [2])
        val_729 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_730 = opset18.Concat(val_6, [10], [4], val_729, [128], axis=0)
        val_732 = opset18.Abs(val_730)
        expand_6 = opset18.Expand(unsqueeze_17, val_732)
        val_736 = opset18.Reshape(add_4, [-1], allowzero=0)
        val_737 = opset18.Concat(val_6, [40], val_736, [128], axis=0)
        _unsafe_view_3 = opset18.Reshape(expand_6, val_737, allowzero=1)
        transpose_9 = opset18.Transpose(_unsafe_view_2, perm=[0, 1, 3, 2])
        matmul_3 = opset18.MatMul(add_720, transpose_9)
        mul_814 = opset18.Mul(matmul_3, 0.088388346)
        val_757 = opset18.Constant(value_ints=[0])
        val_759 = opset18.Constant(value_ints=[-1])
        val_760 = opset18.Reshape(add_4, val_759, allowzero=0)
        val_764 = opset18.Constant(value_ints=[1])
        slice_75 = opset18.Slice(slice_scatter_2, val_757, val_760, [3], val_764)
        add_907 = opset18.Add(mul_814, slice_75)
        val_765 = opset18.Softmax(add_907, axis=-1)
        matmul_4 = opset18.MatMul(val_765, _unsafe_view_3)
        transpose_10 = opset18.Transpose(matmul_4, perm=[0, 2, 1, 3])
        val_770 = opset18.Concat(val_6, val_1, [-1], axis=0)
        view_8 = opset18.Reshape(transpose_10, val_770, allowzero=1)
        val_772 = opset18.Transpose(model_layers_1_self_attn_o_proj_weight, perm=[1, 0])
        linear_5 = opset18.MatMul(view_8, val_772)
        add_950 = opset18.Add(add_592, linear_5)
        pow_4 = opset18.Pow(add_950, 2.0)
        mean_3 = opset18.ReduceMean(pow_4, [-1], noop_with_empty_axes=0, keepdims=1)
        add_963 = opset18.Add(mean_3, 1e-05)
        val_776 = opset18.Sqrt(add_963)
        rsqrt_3 = opset18.Reciprocal(val_776)
        mul_887 = opset18.Mul(add_950, rsqrt_3)
        mul_891 = opset18.Mul(model_layers_1_post_attention_layernorm_weight, mul_887)
        val_777 = opset18.Transpose(model_layers_1_mlp_gate_up_proj_weight, perm=[1, 0])
        linear_6 = opset18.MatMul(mul_891, val_777)
        split_1_split_0, split_1_split_1 = opset18.Split(linear_6, axis=2, num_outputs=2)
        val_778 = opset18.Sigmoid(split_1_split_0)
        silu_1 = opset18.Mul(split_1_split_0, val_778)
        mul_907 = opset18.Mul(split_1_split_1, silu_1)
        val_779 = opset18.Transpose(model_layers_1_mlp_down_proj_weight, perm=[1, 0])
        linear_7 = opset18.MatMul(mul_907, val_779)
        add_1008 = opset18.Add(add_950, linear_7)
        pow_5 = opset18.Pow(add_1008, 2.0)
        mean_4 = opset18.ReduceMean(pow_5, [-1], noop_with_empty_axes=0, keepdims=1)
        add_1021 = opset18.Add(mean_4, 1e-05)
        val_783 = opset18.Sqrt(add_1021)
        rsqrt_4 = opset18.Reciprocal(val_783)
        mul_929 = opset18.Mul(add_1008, rsqrt_4)
        mul_933 = opset18.Mul(model_norm_weight, mul_929)
        val_804 = opset18.Transpose(lm_head_weight, perm=[1, 0])
        linear_8 = opset18.MatMul(mul_933, val_804)
        return linear_8, cat_5, cat_11, cat_6, cat_12

    model = main_graph.to_model_proto(value_infos=value_infos)
    return model


def make_model_with_random_weights():
    model_embed_tokens_weight = numpy.random.rand(100352, 5120).astype(numpy.float32)
    model_layers_0_self_attn_o_proj_weight = numpy.random.rand(5120, 5120).astype(
        numpy.float32
    )
    model_layers_0_self_attn_qkv_proj_weight = numpy.random.rand(7680, 5120).astype(
        numpy.float32
    )
    model_layers_0_mlp_gate_up_proj_weight = numpy.random.rand(35840, 5120).astype(
        numpy.float32
    )
    model_layers_0_mlp_down_proj_weight = numpy.random.rand(5120, 17920).astype(numpy.float32)
    model_layers_0_input_layernorm_weight = numpy.random.rand(5120).astype(numpy.float32)
    model_layers_0_post_attention_layernorm_weight = numpy.random.rand(5120).astype(
        numpy.float32
    )
    model_layers_1_self_attn_o_proj_weight = numpy.random.rand(5120, 5120).astype(
        numpy.float32
    )
    model_layers_1_self_attn_qkv_proj_weight = numpy.random.rand(7680, 5120).astype(
        numpy.float32
    )
    model_layers_1_mlp_gate_up_proj_weight = numpy.random.rand(35840, 5120).astype(
        numpy.float32
    )
    model_layers_1_mlp_down_proj_weight = numpy.random.rand(5120, 17920).astype(numpy.float32)
    model_layers_1_input_layernorm_weight = numpy.random.rand(5120).astype(numpy.float32)
    model_layers_1_post_attention_layernorm_weight = numpy.random.rand(5120).astype(
        numpy.float32
    )
    model_norm_weight = numpy.random.rand(5120).astype(numpy.float32)
    lm_head_weight = numpy.random.rand(100352, 5120).astype(numpy.float32)
    expand_2 = numpy.random.rand(1, 64, 1).astype(numpy.float32)
    model = make_model(
        model_embed_tokens_weight,
        model_layers_0_self_attn_o_proj_weight,
        model_layers_0_self_attn_qkv_proj_weight,
        model_layers_0_mlp_gate_up_proj_weight,
        model_layers_0_mlp_down_proj_weight,
        model_layers_0_input_layernorm_weight,
        model_layers_0_post_attention_layernorm_weight,
        model_layers_1_self_attn_o_proj_weight,
        model_layers_1_self_attn_qkv_proj_weight,
        model_layers_1_mlp_gate_up_proj_weight,
        model_layers_1_mlp_down_proj_weight,
        model_layers_1_input_layernorm_weight,
        model_layers_1_post_attention_layernorm_weight,
        model_norm_weight,
        lm_head_weight,
        expand_2,
    )
    return model


class _Phi4LMTest:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model


def phi4lm_test():
    return _Phi4LMTest()
