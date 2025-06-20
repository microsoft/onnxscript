import numpy

from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT, INT64


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
        arange = opset18.Range(sym_size_int_67, add_4, np.int64(1))
        unsqueeze = opset18.Unsqueeze(arange, [np.int64(0)])
        val_18 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_19 = opset18.Concat(val_1, val_18, axis=0)
        full = opset18.Expand(np.float32(-3.4028235e38), val_19)
        arange_1 = opset18.Range(np.int64(0), add_4, np.int64(1))
        view = opset18.Reshape(arange, [np.int64(-1), np.int64(1)], allowzero=1)
        gt = opset18.Greater(arange_1, view)
        convert_element_type_default = opset18.Cast(gt, to=1)
        mul_14 = opset18.Mul(full, convert_element_type_default)
        unsqueeze_4 = opset18.Unsqueeze(mul_14, [np.int64(0), np.int64(1)])
        val_54 = opset18.Concat(val_6, [np.int64(1)], [np.int64(-1)], [np.int64(-1)], axis=0)
        val_56 = opset18.Abs(val_54)
        expand_1 = opset18.Expand(unsqueeze_4, val_56)
        val_76 = opset18.Constant(value_ints=[0])
        val_78 = opset18.Constant(value_ints=[-1])
        val_79 = opset18.Reshape(add_4, val_78, allowzero=0)
        val_83 = opset18.Constant(value_ints=[1])
        slice_8 = opset18.Slice(expand_1, val_76, val_79, [np.int64(3)], val_83)
        unsqueeze_6 = opset18.Unsqueeze(attention_mask, [np.int64(1), np.int64(2)])
        convert_element_type_default_1 = opset18.Cast(unsqueeze_6, to=1)
        add_86 = opset18.Add(slice_8, convert_element_type_default_1)
        eq_65 = opset18.Equal(add_86, np.float32(0.0))
        val_123 = opset18.Constant(value_ints=[0])
        val_125 = opset18.Constant(value_ints=[-1])
        val_126 = opset18.Reshape(add_4, val_125, allowzero=0)
        val_130 = opset18.Constant(value_ints=[1])
        slice_14 = opset18.Slice(expand_1, val_123, val_126, [np.int64(3)], val_130)
        masked_fill = opset18.Where(eq_65, np.float32(-3.4028235e38), slice_14)
        val_183 = opset18.Shape(expand_1, start=0)
        val_184 = opset18.Gather(val_183, np.int64(2), axis=0)
        val_185 = opset18.Range(np.int64(0), val_184, np.int64(1))
        val_190 = opset18.Unsqueeze(val_185, [np.int64(-1)])
        val_191 = opset18.Transpose(masked_fill, perm=[2, 1, 0, 3])
        val_192 = opset18.Transpose(expand_1, perm=[2, 1, 0, 3])
        val_193 = opset18.ScatterND(val_192, val_190, val_191, reduction="none")
        val_195 = opset18.Shape(expand_1, start=0)
        val_196 = opset18.Gather(val_195, np.int64(1), axis=0)
        val_197 = opset18.Range(np.int64(0), val_196, np.int64(1))
        val_202 = opset18.Unsqueeze(val_197, [np.int64(-1)])
        val_203 = opset18.Transpose(val_193, perm=[1, 2, 0, 3])
        val_204 = opset18.Transpose(expand_1, perm=[1, 0, 2, 3])
        val_205 = opset18.ScatterND(val_204, val_202, val_203, reduction="none")
        slice_scatter_1 = opset18.Transpose(val_205, perm=[1, 0, 2, 3])
        val_207 = opset18.Shape(expand_1, start=0)
        val_208 = opset18.Gather(val_207, np.int64(0), axis=0)
        val_209 = opset18.Range(np.int64(0), val_208, np.int64(1))
        val_214 = opset18.Unsqueeze(val_209, [np.int64(-1)])
        slice_scatter_2 = opset18.ScatterND(
            expand_1, val_214, slice_scatter_1, reduction="none"
        )
        unsqueeze_9 = opset18.Unsqueeze(unsqueeze, [np.int64(1)])
        _to_copy = opset18.Cast(unsqueeze_9, to=1)
        matmul = opset18.MatMul(expand_2, _to_copy)
        transpose = opset18.Transpose(matmul, perm=[0, 2, 1])
        cat = opset18.Concat(transpose, transpose, axis=-1)
        cos = opset18.Cos(cat)
        sin = opset18.Sin(cat)
        pow_1 = opset18.Pow(embedding, np.float32(2.0))
        mean = opset18.ReduceMean(pow_1, [np.int64(-1)], noop_with_empty_axes=0, keepdims=1)
        add_189 = opset18.Add(mean, np.float32(1e-05))
        val_252 = opset18.Sqrt(add_189)
        rsqrt = opset18.Reciprocal(val_252)
        mul_167 = opset18.Mul(embedding, rsqrt)
        mul_171 = opset18.Mul(model_layers_0_input_layernorm_weight, mul_167)
        val_253 = opset18.Transpose(model_layers_0_self_attn_qkv_proj_weight, perm=[1, 0])
        linear = opset18.MatMul(mul_171, val_253)
        val_264 = opset18.Constant(value_ints=[1])
        slice_26 = opset18.Slice(
            linear, [np.int64(0)], [np.int64(5120)], [np.int64(2)], val_264
        )
        val_275 = opset18.Constant(value_ints=[1])
        slice_27 = opset18.Slice(
            linear, [np.int64(5120)], [np.int64(6400)], [np.int64(2)], val_275
        )
        val_285 = opset18.Constant(value_ints=[1])
        slice_28 = opset18.Slice(
            linear, [np.int64(6400)], [np.int64(9223372036854775807)], [np.int64(2)], val_285
        )
        val_291 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_1 = opset18.Reshape(slice_26, val_291, allowzero=1)
        transpose_1 = opset18.Transpose(view_1, perm=[0, 2, 1, 3])
        val_297 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_2 = opset18.Reshape(slice_27, val_297, allowzero=1)
        transpose_2 = opset18.Transpose(view_2, perm=[0, 2, 1, 3])
        val_303 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_3 = opset18.Reshape(slice_28, val_303, allowzero=1)
        transpose_3 = opset18.Transpose(view_3, perm=[0, 2, 1, 3])
        unsqueeze_10 = opset18.Unsqueeze(cos, [np.int64(1)])
        unsqueeze_11 = opset18.Unsqueeze(sin, [np.int64(1)])
        mul_223 = opset18.Mul(transpose_1, unsqueeze_10)
        val_336 = opset18.Constant(value_ints=[1])
        slice_31 = opset18.Slice(
            transpose_1, [np.int64(0)], [np.int64(64)], [np.int64(3)], val_336
        )
        val_346 = opset18.Constant(value_ints=[1])
        slice_32 = opset18.Slice(
            transpose_1,
            [np.int64(64)],
            [np.int64(9223372036854775807)],
            [np.int64(3)],
            val_346,
        )
        neg = opset18.Neg(slice_32)
        cat_1 = opset18.Concat(neg, slice_31, axis=-1)
        mul_240 = opset18.Mul(cat_1, unsqueeze_11)
        add_304 = opset18.Add(mul_223, mul_240)
        mul_252 = opset18.Mul(transpose_2, unsqueeze_10)
        val_356 = opset18.Constant(value_ints=[1])
        slice_33 = opset18.Slice(
            transpose_2, [np.int64(0)], [np.int64(64)], [np.int64(3)], val_356
        )
        val_366 = opset18.Constant(value_ints=[1])
        slice_34 = opset18.Slice(
            transpose_2,
            [np.int64(64)],
            [np.int64(9223372036854775807)],
            [np.int64(3)],
            val_366,
        )
        neg_1 = opset18.Neg(slice_34)
        cat_3 = opset18.Concat(neg_1, slice_33, axis=-1)
        mul_269 = opset18.Mul(cat_3, unsqueeze_11)
        add_345 = opset18.Add(mul_252, mul_269)
        cat_5 = opset18.Concat(past_key_values_key_cache_0, add_345, axis=-2)
        cat_6 = opset18.Concat(past_key_values_value_cache_0, transpose_3, axis=-2)
        unsqueeze_12 = opset18.Unsqueeze(cat_5, [np.int64(2)])
        val_413 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_414 = opset18.Concat(
            val_6, [np.int64(10)], [np.int64(4)], val_413, [np.int64(128)], axis=0
        )
        val_416 = opset18.Abs(val_414)
        expand_3 = opset18.Expand(unsqueeze_12, val_416)
        val_421 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_422 = opset18.Concat(val_6, [np.int64(40)], val_421, [np.int64(128)], axis=0)
        _unsafe_view = opset18.Reshape(expand_3, val_422, allowzero=1)
        unsqueeze_13 = opset18.Unsqueeze(cat_6, [np.int64(2)])
        val_467 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_468 = opset18.Concat(
            val_6, [np.int64(10)], [np.int64(4)], val_467, [np.int64(128)], axis=0
        )
        val_470 = opset18.Abs(val_468)
        expand_4 = opset18.Expand(unsqueeze_13, val_470)
        val_474 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_475 = opset18.Concat(val_6, [np.int64(40)], val_474, [np.int64(128)], axis=0)
        _unsafe_view_1 = opset18.Reshape(expand_4, val_475, allowzero=1)
        transpose_4 = opset18.Transpose(_unsafe_view, perm=[0, 1, 3, 2])
        matmul_1 = opset18.MatMul(add_304, transpose_4)
        mul_433 = opset18.Mul(matmul_1, np.float32(0.088388346))
        val_496 = opset18.Constant(value_ints=[0])
        val_498 = opset18.Constant(value_ints=[-1])
        val_499 = opset18.Reshape(add_4, val_498, allowzero=0)
        val_503 = opset18.Constant(value_ints=[1])
        slice_50 = opset18.Slice(slice_scatter_2, val_496, val_499, [np.int64(3)], val_503)
        add_491 = opset18.Add(mul_433, slice_50)
        val_504 = opset18.Softmax(add_491, axis=-1)
        matmul_2 = opset18.MatMul(val_504, _unsafe_view_1)
        transpose_5 = opset18.Transpose(matmul_2, perm=[0, 2, 1, 3])
        val_509 = opset18.Concat(val_6, val_1, [np.int64(-1)], axis=0)
        view_4 = opset18.Reshape(transpose_5, val_509, allowzero=1)
        val_511 = opset18.Transpose(model_layers_0_self_attn_o_proj_weight, perm=[1, 0])
        linear_1 = opset18.MatMul(view_4, val_511)
        add_534 = opset18.Add(embedding, linear_1)
        pow_2 = opset18.Pow(add_534, np.float32(2.0))
        mean_1 = opset18.ReduceMean(pow_2, [np.int64(-1)], noop_with_empty_axes=0, keepdims=1)
        add_547 = opset18.Add(mean_1, np.float32(1e-05))
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
        pow_3 = opset18.Pow(add_592, np.float32(2.0))
        mean_2 = opset18.ReduceMean(pow_3, [np.int64(-1)], noop_with_empty_axes=0, keepdims=1)
        add_605 = opset18.Add(mean_2, np.float32(1e-05))
        val_523 = opset18.Sqrt(add_605)
        rsqrt_2 = opset18.Reciprocal(val_523)
        mul_548 = opset18.Mul(add_592, rsqrt_2)
        mul_552 = opset18.Mul(model_layers_1_input_layernorm_weight, mul_548)
        val_524 = opset18.Transpose(model_layers_1_self_attn_qkv_proj_weight, perm=[1, 0])
        linear_4 = opset18.MatMul(mul_552, val_524)
        val_534 = opset18.Constant(value_ints=[1])
        slice_51 = opset18.Slice(
            linear_4, [np.int64(0)], [np.int64(5120)], [np.int64(2)], val_534
        )
        val_544 = opset18.Constant(value_ints=[1])
        slice_52 = opset18.Slice(
            linear_4, [np.int64(5120)], [np.int64(6400)], [np.int64(2)], val_544
        )
        val_554 = opset18.Constant(value_ints=[1])
        slice_53 = opset18.Slice(
            linear_4, [np.int64(6400)], [np.int64(9223372036854775807)], [np.int64(2)], val_554
        )
        val_559 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_5 = opset18.Reshape(slice_51, val_559, allowzero=1)
        transpose_6 = opset18.Transpose(view_5, perm=[0, 2, 1, 3])
        val_565 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_6 = opset18.Reshape(slice_52, val_565, allowzero=1)
        transpose_7 = opset18.Transpose(view_6, perm=[0, 2, 1, 3])
        val_571 = opset18.Concat(val_6, val_1, [np.int64(-1)], [np.int64(128)], axis=0)
        view_7 = opset18.Reshape(slice_53, val_571, allowzero=1)
        transpose_8 = opset18.Transpose(view_7, perm=[0, 2, 1, 3])
        unsqueeze_14 = opset18.Unsqueeze(cos, [np.int64(1)])
        unsqueeze_15 = opset18.Unsqueeze(sin, [np.int64(1)])
        mul_604 = opset18.Mul(transpose_6, unsqueeze_14)
        val_602 = opset18.Constant(value_ints=[1])
        slice_56 = opset18.Slice(
            transpose_6, [np.int64(0)], [np.int64(64)], [np.int64(3)], val_602
        )
        val_612 = opset18.Constant(value_ints=[1])
        slice_57 = opset18.Slice(
            transpose_6,
            [np.int64(64)],
            [np.int64(9223372036854775807)],
            [np.int64(3)],
            val_612,
        )
        neg_2 = opset18.Neg(slice_57)
        cat_7 = opset18.Concat(neg_2, slice_56, axis=-1)
        mul_621 = opset18.Mul(cat_7, unsqueeze_15)
        add_720 = opset18.Add(mul_604, mul_621)
        mul_633 = opset18.Mul(transpose_7, unsqueeze_14)
        val_622 = opset18.Constant(value_ints=[1])
        slice_58 = opset18.Slice(
            transpose_7, [np.int64(0)], [np.int64(64)], [np.int64(3)], val_622
        )
        val_632 = opset18.Constant(value_ints=[1])
        slice_59 = opset18.Slice(
            transpose_7,
            [np.int64(64)],
            [np.int64(9223372036854775807)],
            [np.int64(3)],
            val_632,
        )
        neg_3 = opset18.Neg(slice_59)
        cat_9 = opset18.Concat(neg_3, slice_58, axis=-1)
        mul_650 = opset18.Mul(cat_9, unsqueeze_15)
        add_761 = opset18.Add(mul_633, mul_650)
        cat_11 = opset18.Concat(past_key_values_key_cache_1, add_761, axis=-2)
        cat_12 = opset18.Concat(past_key_values_value_cache_1, transpose_8, axis=-2)
        unsqueeze_16 = opset18.Unsqueeze(cat_11, [np.int64(2)])
        val_676 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_677 = opset18.Concat(
            val_6, [np.int64(10)], [np.int64(4)], val_676, [np.int64(128)], axis=0
        )
        val_679 = opset18.Abs(val_677)
        expand_5 = opset18.Expand(unsqueeze_16, val_679)
        val_683 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_684 = opset18.Concat(val_6, [np.int64(40)], val_683, [np.int64(128)], axis=0)
        _unsafe_view_2 = opset18.Reshape(expand_5, val_684, allowzero=1)
        unsqueeze_17 = opset18.Unsqueeze(cat_12, [np.int64(2)])
        val_729 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_730 = opset18.Concat(
            val_6, [np.int64(10)], [np.int64(4)], val_729, [np.int64(128)], axis=0
        )
        val_732 = opset18.Abs(val_730)
        expand_6 = opset18.Expand(unsqueeze_17, val_732)
        val_736 = opset18.Reshape(add_4, [np.int64(-1)], allowzero=0)
        val_737 = opset18.Concat(val_6, [np.int64(40)], val_736, [np.int64(128)], axis=0)
        _unsafe_view_3 = opset18.Reshape(expand_6, val_737, allowzero=1)
        transpose_9 = opset18.Transpose(_unsafe_view_2, perm=[0, 1, 3, 2])
        matmul_3 = opset18.MatMul(add_720, transpose_9)
        mul_814 = opset18.Mul(matmul_3, np.float32(0.088388346))
        val_757 = opset18.Constant(value_ints=[0])
        val_759 = opset18.Constant(value_ints=[-1])
        val_760 = opset18.Reshape(add_4, val_759, allowzero=0)
        val_764 = opset18.Constant(value_ints=[1])
        slice_75 = opset18.Slice(slice_scatter_2, val_757, val_760, [np.int64(3)], val_764)
        add_907 = opset18.Add(mul_814, slice_75)
        val_765 = opset18.Softmax(add_907, axis=-1)
        matmul_4 = opset18.MatMul(val_765, _unsafe_view_3)
        transpose_10 = opset18.Transpose(matmul_4, perm=[0, 2, 1, 3])
        val_770 = opset18.Concat(val_6, val_1, [np.int64(-1)], axis=0)
        view_8 = opset18.Reshape(transpose_10, val_770, allowzero=1)
        val_772 = opset18.Transpose(model_layers_1_self_attn_o_proj_weight, perm=[1, 0])
        linear_5 = opset18.MatMul(view_8, val_772)
        add_950 = opset18.Add(add_592, linear_5)
        pow_4 = opset18.Pow(add_950, np.float32(2.0))
        mean_3 = opset18.ReduceMean(pow_4, [np.int64(-1)], noop_with_empty_axes=0, keepdims=1)
        add_963 = opset18.Add(mean_3, np.float32(1e-05))
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
        pow_5 = opset18.Pow(add_1008, np.float32(2.0))
        mean_4 = opset18.ReduceMean(pow_5, [np.int64(-1)], noop_with_empty_axes=0, keepdims=1)
        add_1021 = opset18.Add(mean_4, np.float32(1e-05))
        val_783 = opset18.Sqrt(add_1021)
        rsqrt_4 = opset18.Reciprocal(val_783)
        mul_929 = opset18.Mul(add_1008, rsqrt_4)
        mul_933 = opset18.Mul(model_norm_weight, mul_929)
        val_804 = opset18.Transpose(lm_head_weight, perm=[1, 0])
        linear_8 = opset18.MatMul(mul_933, val_804)
        return linear_8, cat_5, cat_11, cat_6, cat_12

    model = main_graph.to_model_proto()
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
