# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A one-layer SmolLM model test case, with inputs: input_ids, attention_mask, and position_ids.
This is an onnxscript version of the model.
"""

import numpy
from onnx.helper import make_tensor

import onnxscript.ir as ir
from onnxscript import script
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT, INT64


def make_model(
    input_layernorm_weight_0,
    post_attention_layernorm_weight0,
    norm_weight,
    head_weight,
    self_attn_q_proj_weight0,
    self_attn_k_proj_weight0,
    self_attn_v_proj_weight0,
    self_attn_o_proj_weight0,
    mlp_gate_proj_weight0,
    mlp_up_proj_weight0,
    mlp_down_proj_weight0,
):
    @script()
    def main_graph(
        input0: INT64[1, 10], input1: FLOAT[1, 10], input2: INT64[1, 10]
    ) -> (FLOAT[1, 10, 49152], FLOAT[1, 32, 10, 64], FLOAT[1, 32, 10, 64]):
        model_layers_0_input_layernorm_weight = opset18.Constant(
            value=input_layernorm_weight_0
        )
        model_layers_0_post_attention_layernorm_weight = opset18.Constant(
            value=post_attention_layernorm_weight0
        )
        model_norm_weight = opset18.Constant(value=norm_weight)
        lm_head_weight = opset18.Constant(value=head_weight)
        model_layers_0_self_attn_q_proj_weight = opset18.Constant(
            value=self_attn_q_proj_weight0
        )
        model_layers_0_self_attn_k_proj_weight = opset18.Constant(
            value=self_attn_k_proj_weight0
        )
        model_layers_0_self_attn_v_proj_weight = opset18.Constant(
            value=self_attn_v_proj_weight0
        )
        model_layers_0_self_attn_o_proj_weight = opset18.Constant(
            value=self_attn_o_proj_weight0
        )
        model_layers_0_mlp_gate_proj_weight = opset18.Constant(value=mlp_gate_proj_weight0)
        model_layers_0_mlp_up_proj_weight = opset18.Constant(value=mlp_up_proj_weight0)
        model_layers_0_mlp_down_proj_weight = opset18.Constant(value=mlp_down_proj_weight0)

        embedding = opset18.Gather(lm_head_weight, input0, axis=0)
        minus_inf_10x10 = opset18.ConstantOfShape([10, 10], [-3.4028234663852886e38])
        mask_10x10 = opset18.Trilu(minus_inf_10x10, 1)
        slice_5 = opset18.Reshape(mask_10x10, [1, 1, 10, 10])
        unsqueeze_2 = opset18.Unsqueeze(input1, 1)
        unsqueeze_3 = opset18.Unsqueeze(unsqueeze_2, 2)
        add = slice_5 + unsqueeze_3
        eq = add == 0.0
        slice_10 = slice_5
        masked_fill = opset18.Where(eq, -3.4028235e38, slice_10)
        val_179 = opset18.Transpose(masked_fill, perm=[2, 1, 0, 3])
        slice_scatter = opset18.Transpose(val_179, perm=[2, 1, 0, 3])
        val_191 = opset18.Transpose(slice_scatter, perm=[1, 0, 2, 3])
        slice_scatter_1 = opset18.Transpose(val_191, perm=[1, 0, 2, 3])
        unsqueeze_6 = opset18.Unsqueeze(input2, 1)
        to_copy_1 = opset18.Cast(unsqueeze_6, to=1)
        view_1 = opset18.Constant(
            value=make_tensor(
                "value",
                1,
                dims=[1, 32, 1],
                vals=[
                    1.0,
                    0.7498942017555237,
                    0.5623413324356079,
                    0.4216965138912201,
                    0.3162277638912201,
                    0.23713736236095428,
                    0.17782793939113617,
                    0.1333521455526352,
                    0.10000000149011612,
                    0.07498941570520401,
                    0.05623412877321243,
                    0.04216964915394783,
                    0.03162277862429619,
                    0.0237137358635664,
                    0.017782794311642647,
                    0.01333521492779255,
                    0.009999999776482582,
                    0.007498942315578461,
                    0.005623413249850273,
                    0.0042169648222625256,
                    0.003162277862429619,
                    0.0023713738191872835,
                    0.0017782794311642647,
                    0.0013335214462131262,
                    0.0010000000474974513,
                    0.0007498941849917173,
                    0.000562341301701963,
                    0.00042169648804701865,
                    0.0003162277862429619,
                    0.0002371373848291114,
                    0.00017782794020604342,
                    0.0001333521504420787,
                ],
            )
        )
        view_2 = opset18.Reshape(to_copy_1, [1, 1, 10], allowzero=0)
        bmm = view_1 @ view_2
        view_3 = opset18.Reshape(bmm, [1, 32, 10], allowzero=0)
        transpose = opset18.Transpose(view_3, perm=[0, 2, 1])
        cat = opset18.Concat(transpose, transpose, axis=-1)
        cos = opset18.Cos(cat)
        sin = opset18.Sin(cat)
        pow_1 = embedding**2.0
        mean = opset18.ReduceMean(pow_1, [-1], keepdims=1, noop_with_empty_axes=0)
        add_1 = mean + 1e-05
        val_244 = opset18.Sqrt(add_1)
        rsqrt = opset18.Reciprocal(val_244)
        mul_3 = embedding * rsqrt
        mul_4 = model_layers_0_input_layernorm_weight * mul_3
        t = opset18.Transpose(model_layers_0_self_attn_q_proj_weight, perm=[1, 0])
        view_5 = mul_4 @ t
        t_1 = opset18.Transpose(model_layers_0_self_attn_k_proj_weight, perm=[1, 0])
        view_7 = mul_4 @ t_1
        t_2 = opset18.Transpose(model_layers_0_self_attn_v_proj_weight, perm=[1, 0])
        view_9 = mul_4 @ t_2
        view_10 = opset18.Reshape(view_5, [1, 10, 32, 64], allowzero=0)
        transpose_1 = opset18.Transpose(view_10, perm=[0, 2, 1, 3])
        view_11 = opset18.Reshape(view_7, [1, 10, 32, 64], allowzero=0)
        transpose_2 = opset18.Transpose(view_11, perm=[0, 2, 1, 3])
        view_12 = opset18.Reshape(view_9, [1, 10, 32, 64], allowzero=0)
        transpose_3 = opset18.Transpose(view_12, perm=[0, 2, 1, 3])
        unsqueeze_7 = opset18.Unsqueeze(cos, 1)
        unsqueeze_8 = opset18.Unsqueeze(sin, 1)
        mul_5 = transpose_1 * unsqueeze_7
        val_267 = opset18.Constant(value_ints=[1])
        slice_19 = opset18.Slice(transpose_1, [0], [32], [3], val_267)
        val_277 = opset18.Constant(value_ints=[1])
        slice_20 = opset18.Slice(transpose_1, [32], [9223372036854775807], [3], val_277)
        neg = opset18.Neg(slice_20)
        cat_1 = opset18.Concat(neg, slice_19, axis=-1)
        mul_6 = cat_1 * unsqueeze_8
        add_2 = mul_5 + mul_6
        mul_7 = transpose_2 * unsqueeze_7
        val_287 = opset18.Constant(value_ints=[1])
        slice_21 = opset18.Slice(transpose_2, [0], [32], [3], val_287)
        val_297 = opset18.Constant(value_ints=[1])
        slice_22 = opset18.Slice(transpose_2, [32], [9223372036854775807], [3], val_297)
        neg_1 = opset18.Neg(slice_22)
        cat_2 = opset18.Concat(neg_1, slice_21, axis=-1)
        mul_8 = cat_2 * unsqueeze_8
        add_3 = mul_7 + mul_8
        val_346 = opset18.Reshape(add_3, [-1, 10, 64], allowzero=0)
        val_347 = opset18.Transpose(val_346, perm=[0, 2, 1])
        val_349 = opset18.Reshape(val_347, [1, 32, 64, 10], allowzero=0)
        val_351 = add_2 * [0.35355338]
        val_353 = val_349 * [0.35355338]
        val_354 = val_351 @ val_353
        val_355 = val_354 + slice_scatter_1
        val_356 = opset18.Softmax(val_355, axis=-1)
        getitem = val_356 @ transpose_3
        transpose_4 = opset18.Transpose(getitem, perm=[0, 2, 1, 3])
        view_13 = opset18.Reshape(transpose_4, [1, 10, -1], allowzero=0)
        t_3 = opset18.Transpose(model_layers_0_self_attn_o_proj_weight, perm=[1, 0])
        view_15 = view_13 @ t_3
        add_4 = embedding + view_15
        pow_2 = add_4**2.0
        mean_1 = opset18.ReduceMean(pow_2, [-1], keepdims=1, noop_with_empty_axes=0)
        add_5 = mean_1 + 1e-05
        val_379 = opset18.Sqrt(add_5)
        rsqrt_1 = opset18.Reciprocal(val_379)
        mul_9 = add_4 * rsqrt_1
        mul_10 = model_layers_0_post_attention_layernorm_weight * mul_9
        t_4 = opset18.Transpose(model_layers_0_mlp_gate_proj_weight, perm=[1, 0])
        view_17 = mul_10 @ t_4
        val_383 = opset18.Sigmoid(view_17)
        silu = view_17 * val_383
        t_5 = opset18.Transpose(model_layers_0_mlp_up_proj_weight, perm=[1, 0])
        view_19 = mul_10 @ t_5
        mul_11 = silu * view_19
        t_6 = opset18.Transpose(model_layers_0_mlp_down_proj_weight, perm=[1, 0])
        view_21 = mul_11 @ t_6
        add_6 = add_4 + view_21
        pow_3 = add_6**2.0
        mean_2 = opset18.ReduceMean(pow_3, [-1], keepdims=1, noop_with_empty_axes=0)
        add_7 = mean_2 + 1e-05
        val_391 = opset18.Sqrt(add_7)
        rsqrt_2 = opset18.Reciprocal(val_391)
        mul_12 = add_6 * rsqrt_2
        mul_13 = model_norm_weight * mul_12
        t_7 = opset18.Transpose(lm_head_weight, perm=[1, 0])
        view_23 = mul_13 @ t_7
        to_copy_12 = opset18.Identity(view_23)
        return to_copy_12, add_3, transpose_3

    model = main_graph.to_model_proto()
    return model


def make_model_with_random_weights():
    input_layernorm_weight_0 = numpy.random.rand(2048).astype(numpy.float32)
    post_attention_layernorm_weight0 = numpy.random.rand(2048).astype(numpy.float32)
    norm_weight = numpy.random.rand(2048).astype(numpy.float32)
    head_weight = numpy.random.rand(49152, 2048).astype(numpy.float32)
    self_attn_q_proj_weight0 = numpy.random.rand(2048, 2048).astype(numpy.float32)
    self_attn_k_proj_weight0 = numpy.random.rand(2048, 2048).astype(numpy.float32)
    self_attn_v_proj_weight0 = numpy.random.rand(2048, 2048).astype(numpy.float32)
    self_attn_o_proj_weight0 = numpy.random.rand(2048, 2048).astype(numpy.float32)
    mlp_gate_proj_weight0 = numpy.random.rand(8192, 2048).astype(numpy.float32)
    mlp_up_proj_weight0 = numpy.random.rand(8192, 2048).astype(numpy.float32)
    mlp_down_proj_weight0 = numpy.random.rand(2048, 8192).astype(numpy.float32)
    model = make_model(
        input_layernorm_weight_0,
        post_attention_layernorm_weight0,
        norm_weight,
        head_weight,
        self_attn_q_proj_weight0,
        self_attn_k_proj_weight0,
        self_attn_v_proj_weight0,
        self_attn_o_proj_weight0,
        mlp_gate_proj_weight0,
        mlp_up_proj_weight0,
        mlp_down_proj_weight0,
    )
    return model


class _SmollmTest1:
    def get_onnx_model(self):
        if not hasattr(self, "_onnx_model"):
            model_proto = make_model_with_random_weights()
            model = ir.serde.deserialize_model(model_proto)
            self._onnx_model = model
        return self._onnx_model

    def get_ort_inputs(self):
        if not hasattr(self, "_ort_inputs"):
            inputs = {
                "input0": numpy.random.randint(0, 49152, (1, 10)).astype(numpy.int64),
                "input1": numpy.ones((1, 10), dtype=numpy.float32),
                "input2": numpy.arange(10, dtype=numpy.int64).reshape(1, 10),
            }
            self._ort_inputs = inputs
        return self._ort_inputs


def smollm_test_1():
    return _SmollmTest1()
